"""Generation usage cost estimation."""

from decimal import Decimal

from genai_template.workflow.portfolio.config.models import TokenPricingConfig
from genai_template.workflow.portfolio.domain.generation import TokenUsage

_TOKENS_PER_MILLION = Decimal(1_000_000)


def estimate_generation_cost(
    usage: TokenUsage,
    pricing: TokenPricingConfig,
) -> Decimal | None:
    """Estimate cost only when both provider token counts are available.

    Args:
        usage:
            Complete, partial, or unavailable provider token counts.
        pricing:
            Explicit input and output rates per million tokens.

    Returns:
        Estimated cost, or null when either count is unknown.
    """

    if usage.input_tokens is None or usage.output_tokens is None:
        return None
    input_cost = Decimal(usage.input_tokens) * pricing.input_per_million_tokens
    output_cost = Decimal(usage.output_tokens) * pricing.output_per_million_tokens
    return (input_cost + output_cost) / _TOKENS_PER_MILLION
