"""Response IDs must retain independent records even when legacy names collide."""

import hashlib
import json

import pytest
from easydel.workers.response_store.file_store import FileResponseStore

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("unsafe_first", [False, True])
@pytest.mark.parametrize("delete_unsafe", [False, True])
@pytest.mark.parametrize("compression_level", [0, 3])
def test_colliding_ids_survive_update_reopen_and_delete(tmp_path, unsafe_first, delete_unsafe, compression_level):
    unsafe_id = "../outside"
    safe_id = hashlib.sha256(unsafe_id.encode()).hexdigest()
    ids = [unsafe_id, safe_id] if unsafe_first else [safe_id, unsafe_id]
    store = FileResponseStore(tmp_path, compression_level=compression_level)
    for response_id in ids:
        store.put_response(response_id, {"id": response_id, "turn": 0})
    for response_id in reversed(ids):
        assert store.get_response(response_id) == {"id": response_id, "turn": 0}
        store.put_response(response_id, {"id": response_id, "turn": 1})
    store.close()

    reopened = FileResponseStore(tmp_path, compression_level=compression_level)
    for response_id in ids:
        assert reopened.get_response(response_id) == {"id": response_id, "turn": 1}
    deleted_id, retained_id = (unsafe_id, safe_id) if delete_unsafe else (safe_id, unsafe_id)
    assert reopened.delete_response(deleted_id)
    assert reopened.get_response(deleted_id) is None
    assert reopened.get_response(retained_id) == {"id": retained_id, "turn": 1}
    reopened.close()
    final = FileResponseStore(tmp_path)
    assert final.get_response(retained_id) == {"id": retained_id, "turn": 1}
    assert final.delete_response(retained_id)
    assert list((tmp_path / "responses").iterdir()) == []


@pytest.mark.parametrize("unsafe_first", [False, True])
def test_colliding_ids_eviction_does_not_delete_new_record(tmp_path, unsafe_first):
    unsafe_id = "../outside"
    safe_id = hashlib.sha256(unsafe_id.encode()).hexdigest()
    first, second = (unsafe_id, safe_id) if unsafe_first else (safe_id, unsafe_id)
    store = FileResponseStore(tmp_path, max_stored_responses=1)
    store.put_response(first, {"id": first})
    store.put_response(second, {"id": second})
    assert store.get_response(first) is None
    assert store.get_response(second) == {"id": second}
    store.close()
    reopened = FileResponseStore(tmp_path, max_stored_responses=1)
    assert reopened.get_response(second) == {"id": second}
    assert len(list((tmp_path / "responses").iterdir())) == 1


def test_allocation_checks_suffix_collisions_and_unindexed_files(tmp_path):
    unsafe_id = "../outside"
    digest = hashlib.sha256(unsafe_id.encode()).hexdigest()
    store = FileResponseStore(tmp_path)
    unindexed = store.responses_dir / f"{digest}-2.bin"
    unindexed.write_bytes(b"unindexed legacy record")
    ids = [digest, f"{digest}-1", unsafe_id, f"{digest}-3"]
    for response_id in ids:
        store.put_response(response_id, {"id": response_id})
    store.close()
    reopened = FileResponseStore(tmp_path)
    for response_id in ids:
        assert reopened.get_response(response_id) == {"id": response_id}
    assert unindexed.read_bytes() == b"unindexed legacy record"


@pytest.mark.parametrize("legacy_unsafe", [False, True])
def test_legacy_indexed_filename_remains_readable_and_writable(tmp_path, legacy_unsafe):
    unsafe_id = "../outside"
    safe_id = hashlib.sha256(unsafe_id.encode()).hexdigest()
    legacy_id, new_id = (unsafe_id, safe_id) if legacy_unsafe else (safe_id, unsafe_id)
    responses = tmp_path / "responses"
    responses.mkdir()
    filename = f"{safe_id}.bin"
    (responses / filename).write_text(json.dumps({"id": legacy_id, "turn": 0}))
    (tmp_path / "index.json").write_text(
        json.dumps({"version": 1, "responses": {"items": {legacy_id: {"file": filename}}}})
    )

    store = FileResponseStore(tmp_path)
    assert store.get_response(legacy_id) == {"id": legacy_id, "turn": 0}
    store.put_response(new_id, {"id": new_id, "turn": 1})
    store.put_response(legacy_id, {"id": legacy_id, "turn": 2})
    store.close()
    reopened = FileResponseStore(tmp_path)
    assert reopened.get_response(legacy_id) == {"id": legacy_id, "turn": 2}
    assert reopened.get_response(new_id) == {"id": new_id, "turn": 1}
    assert reopened.delete_response(new_id)
    assert reopened.get_response(legacy_id) == {"id": legacy_id, "turn": 2}
    assert (responses / filename).exists()
