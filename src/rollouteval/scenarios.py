"""Load and validate rollout scenarios from YAML files."""

from pathlib import Path

import yaml

from rollouteval.domain.scenario import Scenario


def load_scenario(path: str | Path) -> Scenario:
    """Load a YAML scenario file and validate its shape and values."""
    with Path(path).open(encoding="utf-8") as scenario_file:
        data = yaml.safe_load(scenario_file)

    if not isinstance(data, dict):
        raise ValueError("scenario document must be a YAML mapping")
    return Scenario.model_validate(data)
