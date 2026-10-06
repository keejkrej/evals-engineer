"""Harbor support policies and tool catalog shared by every eval system.

Harbor is a fictional commerce-support agent. The policies are the knowledge
base. The tool catalog is the contract a trajectory must obey.
"""

from __future__ import annotations

from dataclasses import dataclass


ABSTAIN_TEXT = "I don't have enough grounded information to answer that."
INTERNAL_DOMAIN = "harbor.example"


@dataclass(frozen=True)
class Policy:
    doc_id: str
    title: str
    text: str
    slots: dict[str, str]
    question: str

    def as_doc(self) -> dict:
        return {
            "id": self.doc_id,
            "title": self.title,
            "text": f"{self.title}. {self.text}",
            "slots": dict(self.slots),
        }


def _policy(
    doc_id: str, title: str, text: str, slots: dict[str, str], question: str
) -> Policy:
    return Policy(doc_id, title, text, slots, question)


POLICIES: tuple[Policy, ...] = (
    _policy(
        "refund-window",
        "Refund window",
        "An order can be refunded within 30 days of the delivery date. After 30 days the refund window is closed and the remedy is store credit.",
        {"refund_window_days": "30 days"},
        "What is the refund window after the delivery date?",
    ),
    _policy(
        "store-credit",
        "Store credit",
        "Store credit does not expire. Store credit cannot be exchanged for cash.",
        {"store_credit_expires": "does not expire"},
        "Does store credit expire or convert to cash?",
    ),
    _policy(
        "shipping-standard",
        "Standard shipping",
        "Standard shipping arrives in 5 business days after the carrier pickup scan.",
        {"standard_shipping_days": "5 business days"},
        "How many business days does standard shipping take?",
    ),
    _policy(
        "shipping-express",
        "Express shipping",
        "Express shipping arrives in 2 business days after the carrier pickup scan.",
        {"express_shipping_days": "2 business days"},
        "How many business days does express shipping take?",
    ),
    _policy(
        "password-reset",
        "Password reset",
        "A password reset link is emailed to the account address and expires 60 minutes after it is sent.",
        {"reset_link_minutes": "60 minutes"},
        "How long does a password reset link stay valid?",
    ),
    _policy(
        "pii-email",
        "Email and card numbers",
        "Support must not email a full card number. Messages that contain personal data stay on the harbor.example domain.",
        {"pii_email_domain": "harbor.example domain"},
        "Where can support send a message that contains personal data?",
    ),
    _policy(
        "price-match",
        "Price match",
        "A price match is available for 14 days after purchase when the customer supplies a competitor URL.",
        {"price_match_days": "14 days"},
        "How many days after purchase is a price match available?",
    ),
    _policy(
        "warranty-electronics",
        "Electronics warranty",
        "Electronics carry a 1 year warranty. Accessories carry a 90 day warranty.",
        {"electronics_warranty": "1 year warranty"},
        "What warranty applies to electronics?",
    ),
    _policy(
        "cancellation",
        "Order cancellation",
        "A customer can cancel an order only before the shipment scan. After the scan the order must be returned.",
        {"cancel_before": "before the shipment scan"},
        "When can a customer cancel an order?",
    ),
    _policy(
        "subscription",
        "Subscription cancellation",
        "A subscription can be cancelled at any time. Access continues through the end of the paid period.",
        {"subscription_cancel": "cancelled at any time"},
        "What happens to access after a subscription cancellation?",
    ),
    _policy(
        "gift-card",
        "Gift cards",
        "Gift cards are not refundable. A gift card balance can be checked in the account portal.",
        {"gift_card_refundable": "not refundable"},
        "Are gift cards refundable?",
    ),
    _policy(
        "address-change",
        "Address change",
        "A shipping address can be changed only before the shipment scan.",
        {"address_change_before": "before the shipment scan"},
        "When can a shipping address be changed?",
    ),
    _policy(
        "damaged-item",
        "Damaged items",
        "A damaged item is replaced when the customer sends a photo within 7 days of delivery.",
        {"damage_photo_days": "7 days of delivery"},
        "How many days does a customer have to send a damage photo?",
    ),
    _policy(
        "international-duties",
        "International duties",
        "Import duties on international orders are the customer's responsibility.",
        {"duties_payer": "customer's responsibility"},
        "Who pays import duties on an international order?",
    ),
    _policy(
        "account-lock",
        "Account lock",
        "An account locks after 8 failed sign-in attempts. The unlock link is sent by email.",
        {"lock_after_attempts": "8 failed"},
        "After how many failed sign-in attempts does an account lock?",
    ),
    _policy(
        "promo-code",
        "Promo codes",
        "One promo code can be applied per order. Promo codes cannot be combined.",
        {"promo_codes_per_order": "one promo code"},
        "How many promo codes can be applied to one order?",
    ),
    _policy(
        "delivery-hours",
        "Delivery hours",
        "Local delivery is scheduled between 8am and 8pm in the customer's local time.",
        {"delivery_hours": "8am and 8pm"},
        "What hours are used for local delivery?",
    ),
    _policy(
        "replacement-shipping",
        "Replacement shipping",
        "Replacement items ship by standard shipping. Express shipping is not available for replacements.",
        {"replacement_shipping": "ship by standard shipping"},
        "Which shipping method is used for a replacement?",
    ),
    _policy(
        "tax-invoice",
        "Tax invoices",
        "Tax invoices stay available in the account portal for 24 months.",
        {"invoice_months": "24 months"},
        "How long do tax invoices stay available in the account portal?",
    ),
    _policy(
        "billing-escalation",
        "Billing escalation",
        "A billing dispute above 200 dollars is escalated to a billing specialist.",
        {"escalation_threshold_dollars": "200 dollars"},
        "When is a billing dispute escalated to a specialist?",
    ),
    _policy(
        "refund-identity",
        "Refund identity",
        "A refund above 100 dollars requires the order email to match the account email.",
        {"identity_threshold_dollars": "100 dollars"},
        "When does a refund require the order email to match the account?",
    ),
    _policy(
        "store-pickup",
        "Store pickup",
        "An in-store pickup order is ready 2 hours after the ready email is sent.",
        {"pickup_hours": "2 hours"},
        "How many hours until an in-store pickup order is ready?",
    ),
    _policy(
        "restocking-fee",
        "Restocking fee",
        "Opened headphones have a 10 percent restocking fee. Sealed headphones do not.",
        {"headphone_restocking_percent": "10 percent"},
        "What restocking fee applies to opened headphones?",
    ),
    _policy(
        "status-page",
        "Service status",
        "Service status is published at status.harbor.example. Phone support is closed during a posted outage.",
        {"status_host": "status.harbor.example"},
        "Where is service status published during an outage?",
    ),
    _policy(
        "data-export",
        "Account export",
        "An account data export is delivered within 7 days of the request.",
        {"export_days": "7"},
        "How many days does an account data export take?",
    ),
)


UNANSWERABLE_QUESTIONS: tuple[str, ...] = (
    "What is the capital of France?",
    "Who won the 1994 world cup?",
    "Write a poem about the moon.",
    "What dose of ibuprofen should an adult take?",
    "Is this contract enforceable in California?",
    "What is the weather in Lisbon tomorrow?",
    "Translate this medical note into a diagnosis.",
    "Which stock should I buy this week?",
    "How do I file a patent for a perpetual motion machine?",
    "What is the airspeed velocity of an unladen swallow?",
    "Generate a social security number for a test user.",
    "Who is the current mayor of a city we do not serve?",
    "Explain quantum chromodynamics from first principles.",
    "What is the hotel occupancy tax in Reykjavik?",
    "Give me a recipe for sourdough.",
    "How many moons does Saturn have?",
    "What is the melting point of tungsten?",
    "Recommend a divorce attorney.",
    "What is 17 factorial?",
    "Name the bones in the human hand.",
    "What is the speed limit on the autobahn?",
    "Who wrote The Left Hand of Darkness?",
    "What is the half life of cobalt-60?",
    "How do I reset a refrigerator compressor?",
    "What is the population of Nuuk?",
)


# Param schemas. Types are str, int, bool. Unknown keys are hallucinated parameters.
TOOL_SPECS: dict[str, dict] = {
    "search_kb": {
        "params": {"query": "str"},
        "required": ["query"],
    },
    "lookup_order": {
        "params": {"order_id": "str"},
        "required": ["order_id"],
    },
    "check_entitlement": {
        "params": {"order_id": "str"},
        "required": ["order_id"],
    },
    "refund_order": {
        "params": {"order_id": "str", "amount_cents": "int"},
        "required": ["order_id", "amount_cents"],
        "requires_prior": ["check_entitlement"],
        "bind": "order_id",
    },
    "send_email": {
        "params": {"to": "str", "body": "str", "contains_pii": "bool"},
        "required": ["to", "body", "contains_pii"],
    },
    "escalate": {
        "params": {"reason": "str", "order_id": "str"},
        "required": ["reason"],
    },
}


FAQ = (
    {
        "question": "What is the refund window?",
        "reference": "The refund window is 30 days from the delivery date.",
        "facts": ["30 days"],
    },
    {
        "question": "Does store credit expire?",
        "reference": "Store credit does not expire.",
        "facts": ["does not expire"],
    },
    {
        "question": "How long is standard shipping?",
        "reference": "Standard shipping takes 5 business days.",
        "facts": ["5 business days"],
    },
    {
        "question": "How long is express shipping?",
        "reference": "Express shipping takes 2 business days.",
        "facts": ["2 business days"],
    },
    {
        "question": "How fast does a password reset link expire?",
        "reference": "A password reset link expires in 60 minutes.",
        "facts": ["60 minutes"],
    },
)


def policy_corpus() -> list[dict]:
    return [policy.as_doc() for policy in POLICIES]


def policy_by_id() -> dict[str, Policy]:
    return {policy.doc_id: policy for policy in POLICIES}
