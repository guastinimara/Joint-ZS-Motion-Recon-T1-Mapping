import importlib
import sys
import types

# Only get_prefix is tested; stub mrpro so loading_data can be imported
# even when the full reconstruction stack is not installed.
sys.modules.setdefault("mrpro", types.SimpleNamespace())
loading_data = importlib.import_module("data.loading_data")
get_prefix = loading_data.get_prefix


def test_get_prefix_with_underscores() -> None:
    patient_name = "subj0_meas_example"
    assert get_prefix(patient_name) == "subj0"


def test_get_prefix_without_underscore() -> None:
    patient_name = "patientX"
    assert get_prefix(patient_name) == "patientX"
