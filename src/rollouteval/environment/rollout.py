"""Mutable runtime state initialized from a validated scenario."""

from rollouteval.domain.scenario import InitialEnvironment, RolloutState, Scenario


class MutationLimitExceeded(RuntimeError):
    """Raised when an environment episode has exhausted its mutations."""


class RolloutEnvironment:
    """Own an isolated runtime copy of a scenario's initial environment."""

    def __init__(self, scenario: Scenario) -> None:
        self._scenario_id = scenario.id
        self._baseline = scenario.initial_environment.model_copy(deep=True)
        self._state = self._baseline.model_copy(deep=True)
        self._max_mutations = scenario.max_mutations
        self._mutation_count = 0

    @property
    def scenario_id(self) -> str:
        """Return the stable identity of the scenario initializing this runtime."""
        return self._scenario_id

    @property
    def mutation_count(self) -> int:
        """Return the number of successful mutations in this episode."""
        return self._mutation_count

    def snapshot(self) -> InitialEnvironment:
        """Return an isolated copy of the current runtime state."""
        return self._state.model_copy(deep=True)

    def reset(self) -> None:
        """Restore the runtime state to its original scenario baseline."""
        self._state = self._baseline.model_copy(deep=True)
        self._mutation_count = 0

    def promote(self) -> RolloutState:
        """Promote the candidate to full traffic."""
        return self._mutate_rollout(phase="Promoted", candidate_weight=100)

    def pause(self) -> RolloutState:
        """Pause rollout progression at its current traffic allocation."""
        return self._mutate_rollout(phase="Paused")

    def rollback(self) -> RolloutState:
        """Roll back the candidate to zero traffic."""
        return self._mutate_rollout(phase="RolledBack", candidate_weight=0)

    def _mutate_rollout(
        self, phase: str, candidate_weight: int | None = None
    ) -> RolloutState:
        if self._mutation_count >= self._max_mutations:
            raise MutationLimitExceeded(
                f"mutation limit reached ({self._max_mutations}) for this environment episode"
            )

        rollout_updates = {"phase": phase}
        if candidate_weight is not None:
            rollout_updates["candidate_weight"] = candidate_weight
        updated_rollout = self._state.rollout.model_copy(update=rollout_updates)
        self._state = self._state.model_copy(update={"rollout": updated_rollout})
        self._mutation_count += 1
        return updated_rollout.model_copy(deep=True)
