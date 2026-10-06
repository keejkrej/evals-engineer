"""Counterfactual replay.

Hops before the swapped node keep their recorded outputs. The swapped node
and every hop after it run again. Tool hops call the live tool with arguments
taken from the latest outputs. LLM hops see the updated parent outputs.
"""

from __future__ import annotations

from evals_engineer.report import Metric, Report


def replay(trace: list[dict], swap_id: str, llm, tools: dict) -> dict:
    if not any(hop["id"] == swap_id for hop in trace):
        raise ValueError(f"unknown hop {swap_id}")
    outputs = {}
    replaced = False
    calls = []
    for hop in trace:
        if hop["id"] == swap_id:
            replaced = True
            if hop["kind"] == "tool":
                args = hop["args_fn"](outputs) if "args_fn" in hop else hop.get("args", {})
                outputs[hop["id"]] = tools[hop["name"]](args)
                calls.append({"id": hop["id"], "kind": "tool", "mode": "swapped"})
            else:
                outputs[hop["id"]] = llm(hop, outputs)
                calls.append({"id": hop["id"], "kind": "llm", "mode": "swapped"})
            continue
        if not replaced:
            outputs[hop["id"]] = hop["output"]
            calls.append({"id": hop["id"], "kind": hop["kind"], "mode": "recorded"})
            continue
        if hop["kind"] == "tool":
            args = hop["args_fn"](outputs)
            outputs[hop["id"]] = tools[hop["name"]](args)
            calls.append({"id": hop["id"], "kind": "tool", "mode": "replayed"})
        elif hop["kind"] == "llm":
            outputs[hop["id"]] = llm(hop, outputs)
            calls.append({"id": hop["id"], "kind": "llm", "mode": "replayed"})
        else:
            raise ValueError(f"unknown hop kind {hop['kind']}")
    return {"outputs": outputs, "calls": calls}


def refund_trace(order_id: str = "A1000", days_ago: int = 10, amount_cents: int = 2000) -> list[dict]:
    """A four-hop refund trace. Swap the lookup hop to change the downstream decision."""
    return [
        {
            "id": "plan",
            "kind": "llm",
            "name": "plan",
            "output": {"intent": "refund", "order_id": order_id},
        },
        {
            "id": "lookup",
            "kind": "tool",
            "name": "lookup_order",
            "args": {"order_id": order_id},
            "output": {"order_id": order_id, "status": "delivered", "days_ago": days_ago},
            "args_fn": lambda outputs: {"order_id": outputs["plan"]["order_id"]},
        },
        {
            "id": "decide",
            "kind": "llm",
            "name": "decide",
            "output": {"action": "refund", "order_id": order_id, "amount_cents": amount_cents},
        },
        {
            "id": "act",
            "kind": "tool",
            "name": "refund_order",
            "output": {"refunded": True, "order_id": order_id},
            "args_fn": lambda outputs: outputs["decide"],
        },
    ]


def decision_llm(hop: dict, outputs: dict):
    if hop["name"] == "plan":
        return hop["output"]
    lookup = outputs["lookup"]
    order_id = lookup["order_id"]
    if lookup["days_ago"] <= 30 and lookup["status"] == "delivered":
        return {"action": "refund", "order_id": order_id, "amount_cents": 2000}
    return {"action": "escalate", "order_id": order_id, "reason": "outside the refund window"}


def decision_tools(store: dict):
    def lookup_order(args: dict) -> dict:
        row = dict(store[args["order_id"]])
        row["order_id"] = args["order_id"]
        return row

    def refund_order(args: dict) -> dict:
        if args.get("action") == "escalate":
            return {"refunded": False, "escalated": True, "order_id": args["order_id"]}
        return {"refunded": True, "escalated": False, "order_id": args["order_id"], "amount_cents": args["amount_cents"]}

    return {"lookup_order": lookup_order, "refund_order": refund_order}


def replay_report(result: dict) -> Report:
    final = result["outputs"].get("act", {})
    return Report(
        title="Counterfactual replay",
        passed="act" in result["outputs"],
        summary=f"Replayed {len(result['calls'])} hops. Final action refunded={final.get('refunded')}.",
        metrics=[Metric("hops", float(len(result["calls"])))],
        lines=[f"{call['id']}: {call['mode']}" for call in result["calls"]],
    )
