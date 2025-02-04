from pathlib import Path

import pytest
from ruamel.yaml import YAML

from bioimageio.spec.utils.transformers import load_and_resolve_spec

yaml = YAML(typ="safe")


@pytest.fixture
def rf_config_path_v0_1():
    return Path(__file__).parent / "../specs/models/sklearn/RandomForestClassifier_v0_1.model.yaml"


@pytest.fixture
def rf_config_path_v0_3_0():
    return Path(__file__).parent / "../specs/models/sklearn/RandomForestClassifier.model.yaml"


rf_config_path = rf_config_path_v0_3_0


@pytest.fixture
def rf_resolved_spec(rf_config_path):
    return load_and_resolve_spec(rf_config_path)
