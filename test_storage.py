import json

from storage import atomic_write_json, load_json


def test_atomic_write_json_round_trip_preserves_unicode(tmp_path):
    path = tmp_path / "memories.json"
    data = {"name": "记忆", "items": ["中文", {"score": 1}]}

    atomic_write_json(str(path), data)

    assert load_json(str(path), {}) == data
    assert "记忆" in path.read_text(encoding="utf-8")


def test_load_json_recovers_from_backup_when_primary_is_corrupt(tmp_path):
    path = tmp_path / "memories.json"
    backup_data = {"version": "backup", "items": [1]}
    current_data = {"version": "current", "items": [2]}

    atomic_write_json(str(path), backup_data)
    atomic_write_json(str(path), current_data)
    path.write_text("{ broken", encoding="utf-8")

    assert load_json(str(path), {"default": True}) == backup_data
    corrupt_files = list(tmp_path.glob("memories.json.corrupt.*"))
    assert len(corrupt_files) == 1
    assert corrupt_files[0].read_text(encoding="utf-8") == "{ broken"


def test_load_json_returns_default_when_primary_and_backup_are_corrupt(tmp_path):
    path = tmp_path / "memories.json"
    default = {"safe": []}
    path.write_text("{ broken", encoding="utf-8")
    path.with_suffix(path.suffix + ".bak").write_text("{ also broken", encoding="utf-8")

    assert load_json(str(path), default) == default
    assert len(list(tmp_path.glob("memories.json.corrupt.*"))) == 1


def test_atomic_write_json_first_write_does_not_create_backup(tmp_path):
    path = tmp_path / "memories.json"

    atomic_write_json(str(path), {"first": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {"first": True}
    assert not path.with_suffix(path.suffix + ".bak").exists()


def test_atomic_write_json_removes_temp_file_after_success(tmp_path):
    path = tmp_path / "memories.json"

    atomic_write_json(str(path), {"ok": True})

    assert list(tmp_path.glob("memories.json.tmp.*")) == []
