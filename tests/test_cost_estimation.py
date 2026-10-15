from decimal import Decimal

import pytest
from pydantic import ValidationError

from rollouteval.agent import ModelUsage
from rollouteval.observability import (
    CostEstimationUnavailable,
    CostEstimator,
    ModelCallCost,
    ModelTokenPricing,
    PricingSnapshot,
    RunCostEstimate,
    RunTrace,
    TraceEvent,
    TraceEventType,
)
from rollouteval.agent import AgentToolName

RATES = {
    "input_per_million_usd": Decimal("1"),
    "cached_input_per_million_usd": Decimal("0.10"),
    "cache_write_per_million_usd": Decimal("1.25"),
    "output_per_million_usd": Decimal("5"),
}


def _pricing(model: str = "model-x", **overrides) -> ModelTokenPricing:
    return ModelTokenPricing(model=model, **(RATES | overrides))


def _snapshot(*models: ModelTokenPricing, pricing_id: str = "test-pricing-2026-01") -> PricingSnapshot:
    return PricingSnapshot(pricing_id=pricing_id, models=models or (_pricing(),))


def _usage(
    *,
    input_tokens: int = 100,
    output_tokens: int = 20,
    total_tokens: int = 120,
    cached_input_tokens: int | None = 20,
    cache_write_tokens: int | None = 10,
    reasoning_tokens: int | None = 8,
) -> ModelUsage:
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_write_tokens=cache_write_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def _event(
    sequence: int,
    event_type: TraceEventType,
    *,
    step: int | None = None,
    model: str | None = None,
    usage: ModelUsage | None = None,
    tool_name: AgentToolName | None = None,
) -> TraceEvent:
    return TraceEvent(
        sequence=sequence,
        event_type=event_type,
        step=step,
        model=model,
        model_usage=usage,
        tool_name=tool_name,
    )


def _trace(*events: TraceEvent) -> RunTrace:
    return RunTrace(run_id="run-cost", scenario_id="scenario", events=events)


@pytest.mark.parametrize(
    "field",
    [
        "input_per_million_usd",
        "cached_input_per_million_usd",
        "cache_write_per_million_usd",
        "output_per_million_usd",
    ],
)
def test_model_token_pricing_rejects_negative_rates(field: str) -> None:
    with pytest.raises(ValidationError):
        _pricing(**{field: Decimal("-0.01")})


def test_model_token_pricing_accepts_decimal_rates_and_is_immutable() -> None:
    pricing = _pricing()
    assert pricing.input_per_million_usd == Decimal("1")
    with pytest.raises(ValidationError):
        pricing.model = "other"


def test_model_token_pricing_rejects_blank_model_and_binary_float_rate() -> None:
    with pytest.raises(ValidationError):
        _pricing("   ")
    with pytest.raises(ValidationError):
        _pricing(input_per_million_usd=1.0)


def test_pricing_snapshot_validates_identity_and_unique_exact_models() -> None:
    with pytest.raises(ValidationError):
        PricingSnapshot(pricing_id=" ", models=(_pricing(),))
    with pytest.raises(ValidationError, match="unique"):
        _snapshot(_pricing(), _pricing())
    snapshot = _snapshot(_pricing("model-x"), _pricing("model-y"))
    assert snapshot.pricing_for("model-x").model == "model-x"
    with pytest.raises(CostEstimationUnavailable, match="exact model identifier"):
        snapshot.pricing_for("model-x-latest")


def test_pricing_snapshot_is_frozen_and_copy_safe() -> None:
    source = [_pricing()]
    snapshot = PricingSnapshot(pricing_id="rates", models=source)
    source.append(_pricing("later"))

    assert len(snapshot.models) == 1
    with pytest.raises(ValidationError):
        snapshot.pricing_id = "changed"


def test_one_call_uses_exact_category_costs_without_double_billing() -> None:
    usage = _usage()
    trace = _trace(_event(0, TraceEventType.MODEL_COMPLETED, step=1, model="model-x", usage=usage))

    estimate = CostEstimator().estimate(trace, _snapshot())
    call = estimate.calls[0]

    assert call.ordinary_input_tokens == 70
    assert call.cached_input_tokens == 20
    assert call.cache_write_tokens == 10
    assert call.output_tokens == 20
    assert call.ordinary_input_cost_usd == Decimal("0.00007")
    assert call.cached_input_cost_usd == Decimal("0.000002")
    assert call.cache_write_cost_usd == Decimal("0.0000125")
    assert call.output_cost_usd == Decimal("0.0001")
    assert call.total_cost_usd == Decimal("0.0001845")
    assert estimate.total_cost_usd == Decimal("0.0001845")
    assert estimate.pricing_id == "test-pricing-2026-01"


def test_total_and_reasoning_tokens_are_informational_not_extra_charges() -> None:
    ordinary = _usage(total_tokens=120, reasoning_tokens=8)
    changed_breakdowns = _usage(total_tokens=900, reasoning_tokens=90)
    estimator = CostEstimator()
    first = estimator.estimate(
        _trace(_event(0, TraceEventType.MODEL_COMPLETED, step=1, model="model-x", usage=ordinary)),
        _snapshot(),
    )
    second = estimator.estimate(
        _trace(
            _event(
                0,
                TraceEventType.MODEL_COMPLETED,
                step=1,
                model="model-x",
                usage=changed_breakdowns,
            )
        ),
        _snapshot(),
    )
    assert first.total_cost_usd == second.total_cost_usd


@pytest.mark.parametrize(
    ("cached", "write", "expected_ordinary"),
    [(100, 0, 0), (0, 100, 0), (0, 0, 100)],
)
def test_input_tokens_are_partitioned_without_double_charging(
    cached: int, write: int, expected_ordinary: int
) -> None:
    usage = _usage(
        input_tokens=100,
        output_tokens=0,
        total_tokens=100,
        cached_input_tokens=cached,
        cache_write_tokens=write,
        reasoning_tokens=0,
    )
    estimate = CostEstimator().estimate(
        _trace(_event(0, TraceEventType.MODEL_COMPLETED, step=1, model="model-x", usage=usage)),
        _snapshot(),
    )
    assert estimate.calls[0].ordinary_input_tokens == expected_ordinary


@pytest.mark.parametrize("field", ["cached_input_tokens", "cache_write_tokens"])
def test_unknown_input_category_makes_exact_cost_unavailable(field: str) -> None:
    usage = _usage(**{field: None})
    trace = _trace(_event(0, TraceEventType.MODEL_COMPLETED, step=1, model="model-x", usage=usage))

    with pytest.raises(CostEstimationUnavailable, match="unknown"):
        CostEstimator().estimate(trace, _snapshot())


def test_explicit_zero_input_category_is_known_and_priceable() -> None:
    usage = _usage(cached_input_tokens=0, cache_write_tokens=0)
    estimate = CostEstimator().estimate(
        _trace(_event(0, TraceEventType.MODEL_COMPLETED, step=1, model="model-x", usage=usage)),
        _snapshot(),
    )
    assert estimate.calls[0].ordinary_input_tokens == 100
    assert estimate.calls[0].cached_input_tokens == 0
    assert estimate.calls[0].cache_write_tokens == 0


def test_invalid_input_breakdown_fails_without_negative_ordinary_tokens() -> None:
    usage = _usage(input_tokens=100, cached_input_tokens=80, cache_write_tokens=30)
    trace = _trace(_event(0, TraceEventType.MODEL_COMPLETED, step=1, model="model-x", usage=usage))

    with pytest.raises(CostEstimationUnavailable, match="exceeding input_tokens"):
        CostEstimator().estimate(trace, _snapshot())


def test_calls_remain_in_trace_order_and_only_completed_models_are_costed() -> None:
    usage1 = _usage(input_tokens=10, output_tokens=2, total_tokens=12, cached_input_tokens=0, cache_write_tokens=0)
    usage2 = _usage(input_tokens=20, output_tokens=3, total_tokens=23, cached_input_tokens=1, cache_write_tokens=2)
    trace = _trace(
        _event(0, TraceEventType.RUN_STARTED),
        _event(1, TraceEventType.MODEL_COMPLETED, step=1, model="model-x", usage=usage1),
        _event(
            2,
            TraceEventType.TOOL_COMPLETED,
            step=1,
            tool_name=AgentToolName.GET_ROLLOUT_STATE,
        ),
        _event(3, TraceEventType.MODEL_FAILED, step=2),
        _event(4, TraceEventType.MODEL_COMPLETED, step=3, model="model-y", usage=usage2),
        _event(5, TraceEventType.RUN_COMPLETED),
    )
    pricing = _snapshot(_pricing("model-x"), _pricing("model-y", output_per_million_usd=Decimal("2")))

    estimate = CostEstimator().estimate(trace, pricing)

    assert [(call.step, call.model) for call in estimate.calls] == [(1, "model-x"), (3, "model-y")]
    assert estimate.total_cost_usd == sum((call.total_cost_usd for call in estimate.calls), Decimal("0"))


def test_empty_or_failed_only_trace_has_zero_cost() -> None:
    estimator = CostEstimator()
    for trace in (_trace(), _trace(_event(0, TraceEventType.MODEL_FAILED, step=1))):
        estimate = estimator.estimate(trace, _snapshot())
        assert estimate.calls == ()
        assert estimate.total_cost_usd == Decimal("0")


@pytest.mark.parametrize(
    ("model", "usage", "message"),
    [(None, _usage(), "model identifier"), ("model-x", None, "model usage")],
)
def test_completed_model_without_required_metadata_fails(model, usage, message: str) -> None:
    trace = _trace(_event(0, TraceEventType.MODEL_COMPLETED, step=1, model=model, usage=usage))
    with pytest.raises(CostEstimationUnavailable, match=message):
        CostEstimator().estimate(trace, _snapshot())


def test_missing_pricing_for_any_model_fails_whole_estimate() -> None:
    usage = _usage()
    trace = _trace(
        _event(0, TraceEventType.MODEL_COMPLETED, step=1, model="model-x", usage=usage),
        _event(1, TraceEventType.MODEL_COMPLETED, step=2, model="model-y", usage=usage),
    )
    with pytest.raises(CostEstimationUnavailable, match="model-y"):
        CostEstimator().estimate(trace, _snapshot())


def test_different_models_use_their_own_exact_rates() -> None:
    usage = _usage()
    trace = _trace(
        _event(0, TraceEventType.MODEL_COMPLETED, step=1, model="model-x", usage=usage),
        _event(1, TraceEventType.MODEL_COMPLETED, step=2, model="model-y", usage=usage),
    )
    pricing = _snapshot(
        _pricing("model-x"),
        _pricing(
            "model-y",
            input_per_million_usd=Decimal("2"),
            cached_input_per_million_usd=Decimal("0.20"),
            cache_write_per_million_usd=Decimal("2.50"),
            output_per_million_usd=Decimal("10"),
        ),
    )

    estimate = CostEstimator().estimate(trace, pricing)

    assert estimate.calls[1].total_cost_usd == estimate.calls[0].total_cost_usd * 2


def test_same_trace_can_be_repriced_without_mutating_history() -> None:
    usage = _usage()
    trace = _trace(_event(0, TraceEventType.MODEL_COMPLETED, step=1, model="model-x", usage=usage))
    original_trace = trace.model_copy(deep=True)
    first = CostEstimator().estimate(trace, _snapshot(pricing_id="rates-a"))
    second_pricing = _snapshot(
        _pricing(
            input_per_million_usd=Decimal("2"),
            cached_input_per_million_usd=Decimal("0.20"),
            cache_write_per_million_usd=Decimal("2.50"),
            output_per_million_usd=Decimal("10"),
        ),
        pricing_id="rates-b",
    )
    second = CostEstimator().estimate(trace, second_pricing)

    assert first.pricing_id == "rates-a"
    assert second.pricing_id == "rates-b"
    assert second.total_cost_usd == first.total_cost_usd * 2
    assert trace == original_trace
    assert trace.events[0].model_usage == usage
    assert second_pricing == _snapshot(
        _pricing(
            input_per_million_usd=Decimal("2"),
            cached_input_per_million_usd=Decimal("0.20"),
            cache_write_per_million_usd=Decimal("2.50"),
            output_per_million_usd=Decimal("10"),
        ),
        pricing_id="rates-b",
    )


def test_cost_contracts_are_immutable_and_contain_no_scenario_or_client() -> None:
    trace = _trace(
        _event(0, TraceEventType.MODEL_COMPLETED, step=1, model="model-x", usage=_usage())
    )
    estimate = CostEstimator().estimate(trace, _snapshot())
    pricing = _pricing()
    usage = _usage()

    assert set(RunCostEstimate.model_fields) == {"pricing_id", "calls", "total_cost_usd", "currency"}
    assert "usage" not in ModelCallCost.model_fields
    assert not {"scenario", "expected_decision", "client", "api_key"} & (
        set(ModelTokenPricing.model_fields)
        | set(PricingSnapshot.model_fields)
        | set(ModelCallCost.model_fields)
        | set(RunCostEstimate.model_fields)
    )
    with pytest.raises(ValidationError):
        estimate.pricing_id = "changed"
    with pytest.raises(ValidationError):
        estimate.calls[0].total_cost_usd = Decimal("0")
    with pytest.raises(ValidationError):
        pricing.model = "changed"
    with pytest.raises(ValidationError):
        usage.input_tokens = 0
