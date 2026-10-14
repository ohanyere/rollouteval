"""Mutable runtime state initialized from a validated scenario."""

from rollouteval.domain.scenario import InitialEnvironment, Scenario


class RolloutEnvironment:
    """Own an isolated runtime copy of a scenario's initial environment."""

    def __init__(self, scenario: Scenario) -> None:
        self._baseline = scenario.initial_environment.model_copy(deep=True)
        self._state = self._baseline.model_copy(deep=True)

    def snapshot(self) -> InitialEnvironment:
        """Return an isolated copy of the current runtime state."""
        return self._state.model_copy(deep=True)

    def reset(self) -> None:
        """Restore the runtime state to its original scenario baseline."""
        self._state = self._baseline.model_copy(deep=True)
