"""Controlled read-only observations of a rollout environment."""

from pydantic import BaseModel, ConfigDict, model_validator

from rollouteval.domain.scenario import RolloutState, RuntimeEvent, ServiceMetrics
from rollouteval.environment import RolloutEnvironment


class ServiceMetricsObservation(BaseModel):
    """Represent available service metrics or their explicit absence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    available: bool
    metrics: ServiceMetrics | None
    message: str | None = None

    @model_validator(mode="after")
    def validate_availability(self) -> "ServiceMetricsObservation":
        if self.available and self.metrics is None:
            raise ValueError("available service metrics must include metrics")
        if not self.available and self.metrics is not None:
            raise ValueError("unavailable service metrics cannot include metrics")
        if not self.available and not self.message:
            raise ValueError("unavailable service metrics must include an explanation")
        return self


class RolloutObservationTools:
    """Expose only rollout state, service metrics, and runtime events."""

    def __init__(self, environment: RolloutEnvironment) -> None:
        self._environment = environment

    def get_rollout_state(self) -> RolloutState:
        """Return an isolated copy of the observable rollout state."""
        return self._environment.snapshot().rollout.model_copy(deep=True)

    def get_service_metrics(self) -> ServiceMetricsObservation:
        """Return service metrics or an explicit unavailable observation."""
        metrics = self._environment.snapshot().metrics
        if metrics is None:
            return ServiceMetricsObservation(
                available=False,
                metrics=None,
                message="Service metrics are unavailable.",
            )
        return ServiceMetricsObservation(available=True, metrics=metrics.model_copy(deep=True))

    def get_recent_events(self) -> tuple[RuntimeEvent, ...]:
        """Return isolated copies of all runtime events, preserving recency."""
        return tuple(event.model_copy(deep=True) for event in self._environment.snapshot().events)


class RolloutMutationTools:
    """Delegate the three supported rollout mutations to the environment."""

    def __init__(self, environment: RolloutEnvironment) -> None:
        self._environment = environment

    def promote_rollout(self) -> RolloutState:
        """Promote the candidate and return the resulting rollout state."""
        return self._environment.promote()

    def pause_rollout(self) -> RolloutState:
        """Pause rollout progression and return the resulting rollout state."""
        return self._environment.pause()

    def rollback_rollout(self) -> RolloutState:
        """Roll back the candidate and return the resulting rollout state."""
        return self._environment.rollback()
