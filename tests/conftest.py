# ruff: noqa: E402

"""Bootstrap de imports (src/ en sys.path) y fixtures compartidas entre módulos de test.

pytest carga conftest.py antes de recolectar los tests del directorio, así que el bootstrap de
sys.path vive UNA sola vez aquí en lugar de repetirse en cada archivo de test.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from data_prep import build_long_dataset, load_raw
from features import build_country_features


@pytest.fixture(scope="session")
def raw():
    return load_raw()


@pytest.fixture(scope="session")
def long_df():
    return build_long_dataset(save_to=None)


@pytest.fixture(scope="session")
def feats(long_df):
    return build_country_features(long_df)
