import importlib
import sys
import types

# Only get_prefix is tested; stub mrpro so loading_data can be imported
# even when the full reconstruction stack is not installed.
sys.modules.setdefault("mrpro", types.SimpleNamespace())
loading_data = importlib.import_module("data.loading_data")
get_prefix = loading_data.get_prefix


def test_get_prefix_with_underscores() -> None:
    patient_name = "vol2_meas_MID82_CV_gc_13x13x8_series"
    assert get_prefix(patient_name) == "vol2"


def test_get_prefix_without_underscore() -> None:
    patient_name = "patientX"
    assert get_prefix(patient_name) == "patientX"
