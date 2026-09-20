import shutil
import tempfile
from pathlib import Path

from neova.api.store import Store, proposal_id
from neova.config import DATA_DIR


def test_state_survives_restart():
    with tempfile.TemporaryDirectory() as root:
        shutil.copy(DATA_DIR / "neova_data.json", Path(root) / "neova_data.json")

        first = Store(data_dir=Path(root))
        with first.lock:
            first.data["appointments"].append({"appointment_id": "APT-TEST", "customer_id": "NEO-88213",
                                               "slot_id": "SLOT-7A31",
                                               "start": "2026-08-27T09:00:00+02:00",
                                               "end": "2026-08-27T11:00:00+02:00",
                                               "reason": "no_internet", "override_reason": None,
                                               "created_at": "2026-08-25T10:00:00+02:00"})
            first.idempotency_keys["appointments:k"] = {"hash": "h", "response": {"appointment_id": "APT-TEST"}}
            first.proposals["p"] = {"customer_id": "NEO-88213"}
            first.write()

        second = Store(data_dir=Path(root))
        assert second.get_appointment_by_id("APT-TEST") is not None
        assert second.idempotency_keys["appointments:k"]["response"]["appointment_id"] == "APT-TEST"
        assert second.proposals["p"]["customer_id"] == "NEO-88213"


def test_fresh_store_uses_seed_when_no_runtime_state():
    with tempfile.TemporaryDirectory() as root:
        shutil.copy(DATA_DIR / "neova_data.json", Path(root) / "neova_data.json")
        store = Store(data_dir=Path(root))
        assert store.data["appointments"] == []


def test_proposal_id_covers_override_reason():
    base = ("NEO-88213", "SLOT-7A31", "no_internet")
    assert proposal_id(*base, None, "notice") != proposal_id(*base, "pto_damaged", "notice")
    assert proposal_id(*base, None, "notice") == proposal_id(*base, None, "notice")
