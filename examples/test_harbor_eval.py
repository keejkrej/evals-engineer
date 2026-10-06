"""Example pytest run against a JSONL eval set.

    py -3 -m pytest examples/test_harbor_eval.py -p evals_engineer.plugin --eval-dataset data/golden/v1/cases.jsonl -q
"""


def test_case(eval_case, eval_recorder):
    verdict = eval_case["expected"]["verdict"]
    recorded = eval_case["annotators"]["ada"]["verdict"]
    passed = verdict == recorded
    eval_recorder(case_id=eval_case["id"], correct=passed, stratum=eval_case["stratum"])
    assert passed
