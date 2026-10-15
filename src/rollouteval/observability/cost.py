"""Deterministic cost estimates from trace usage and caller-supplied rates."""

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rollouteval.agent.contracts import ModelUsage
from rollouteval.observability.trace import RunTrace, TraceEventType

_TOKENS_PER_MILLION = Decimal("1000000")
_ZERO = Decimal("0")


class CostEstimationUnavailable(ValueError):
    """Raised when trace or pricing data is insufficient for an exact estimate."""


class ModelTokenPricing(BaseModel):
    """Per-million-token USD rates for one exact model identifier."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    model: str = Field(min_length=1)
    input_per_million_usd: Decimal = Field(strict=True, ge=_ZERO)
    cached_input_per_million_usd: Decimal = Field(strict=True, ge=_ZERO)
    cache_write_per_million_usd: Decimal = Field(strict=True, ge=_ZERO)
    output_per_million_usd: Decimal = Field(strict=True, ge=_ZERO)


class PricingSnapshot(BaseModel):
    """Immutable, version-identified schedule of caller-provided model rates."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    pricing_id: str = Field(min_length=1)
    models: tuple[ModelTokenPricing, ...]

    @model_validator(mode="after")
    def validate_unique_models(self) -> "PricingSnapshot":
        identifiers = [pricing.model for pricing in self.models]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("model identifiers must be unique within a pricing snapshot")
        return self

    def pricing_for(self, model: str) -> ModelTokenPricing:
        """Return an exact identifier match or raise a clear lookup error."""
        for pricing in self.models:
            if pricing.model == model:
                return pricing
        raise CostEstimationUnavailable(
            f"no pricing for exact model identifier {model!r} in {self.pricing_id!r}"
        )


class ModelCallCost(BaseModel):
    """Auditable token categories and exact USD costs for one model call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step: int = Field(ge=1)
    model: str = Field(min_length=1)
    ordinary_input_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    cache_write_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    ordinary_input_cost_usd: Decimal = Field(strict=True, ge=_ZERO)
    cached_input_cost_usd: Decimal = Field(strict=True, ge=_ZERO)
    cache_write_cost_usd: Decimal = Field(strict=True, ge=_ZERO)
    output_cost_usd: Decimal = Field(strict=True, ge=_ZERO)
    total_cost_usd: Decimal = Field(strict=True, ge=_ZERO)


class RunCostEstimate(BaseModel):
    """Per-call cost breakdown and total under one pricing snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    pricing_id: str = Field(min_length=1)
    calls: tuple[ModelCallCost, ...]
    total_cost_usd: Decimal = Field(strict=True, ge=_ZERO)
    currency: Literal["USD"] = "USD"


class CostEstimator:
    """Estimate model-call cost using only trace facts and supplied prices."""

    def estimate(self, trace: RunTrace, pricing: PricingSnapshot) -> RunCostEstimate:
        """Price completed model calls in their original trace order."""
        calls: list[ModelCallCost] = []
        for event in trace.events:
            if event.event_type is not TraceEventType.MODEL_COMPLETED:
                continue
            if event.step is None:
                raise CostEstimationUnavailable(
                    "MODEL_COMPLETED event is missing its model-output step"
                )
            if event.model is None:
                raise CostEstimationUnavailable("MODEL_COMPLETED event is missing its model identifier")
            if event.model_usage is None:
                raise CostEstimationUnavailable("MODEL_COMPLETED event is missing model usage")
            calls.append(self._estimate_call(event.step, event.model, event.model_usage, pricing))

        call_tuple = tuple(calls)
        return RunCostEstimate(
            pricing_id=pricing.pricing_id,
            calls=call_tuple,
            total_cost_usd=sum((call.total_cost_usd for call in call_tuple), _ZERO),
        )

    @staticmethod
    def _estimate_call(
        step: int,
        model: str,
        usage: ModelUsage,
        pricing: PricingSnapshot,
    ) -> ModelCallCost:
        if usage.cached_input_tokens is None:
            raise CostEstimationUnavailable(
                f"model {model!r} at step {step} has unknown cached_input_tokens"
            )
        if usage.cache_write_tokens is None:
            raise CostEstimationUnavailable(
                f"model {model!r} at step {step} has unknown cache_write_tokens"
            )
        if usage.cached_input_tokens + usage.cache_write_tokens > usage.input_tokens:
            raise CostEstimationUnavailable(
                f"model {model!r} at step {step} has input categories exceeding input_tokens"
            )

        rates = pricing.pricing_for(model)
        ordinary_tokens = usage.input_tokens - usage.cached_input_tokens - usage.cache_write_tokens
        ordinary_cost = ordinary_tokens * rates.input_per_million_usd / _TOKENS_PER_MILLION
        cached_cost = usage.cached_input_tokens * rates.cached_input_per_million_usd / _TOKENS_PER_MILLION
        write_cost = usage.cache_write_tokens * rates.cache_write_per_million_usd / _TOKENS_PER_MILLION
        output_cost = usage.output_tokens * rates.output_per_million_usd / _TOKENS_PER_MILLION
        total_cost = ordinary_cost + cached_cost + write_cost + output_cost
        return ModelCallCost(
            step=step,
            model=model,
            ordinary_input_tokens=ordinary_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            output_tokens=usage.output_tokens,
            ordinary_input_cost_usd=ordinary_cost,
            cached_input_cost_usd=cached_cost,
            cache_write_cost_usd=write_cost,
            output_cost_usd=output_cost,
            total_cost_usd=total_cost,
        )
