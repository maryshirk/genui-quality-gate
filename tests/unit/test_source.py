import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from genui_quality_gate.queue import QueueUnavailable
from genui_quality_gate.source import load_checkpoint, publish_file
from genui_quality_gate.tasks import TaskPayload, message_identity
from tests.unit.test_inference import PAYLOAD


async def test_source_retries_same_task_and_checkpoints_only_after_confirm(tmp_path: Path) -> None:
    source, checkpoint = tmp_path / "input.jsonl", tmp_path / "checkpoint.json"
    source.write_text(json.dumps(PAYLOAD) + "\n" + json.dumps(PAYLOAD) + "\n")
    publisher = AsyncMock()
    publisher.publish.side_effect = [QueueUnavailable(), QueueUnavailable(), None, None]
    with patch("genui_quality_gate.source.asyncio.sleep", new_callable=AsyncMock) as sleep:
        result = await publish_file(source, checkpoint, publisher)
    calls = publisher.publish.call_args_list
    assert calls[0].args[0].task_id == calls[2].args[0].task_id
    assert calls[2].args[0].task_id != calls[3].args[0].task_id
    assert result == {"accepted": 2, "rejected": 2}
    assert sleep.await_count == 2
    assert sleep.await_args_list[1].args[0] > sleep.await_args_list[0].args[0]
    publisher.reset_mock()
    await publish_file(source, checkpoint, publisher)
    publisher.publish.assert_not_called()


async def test_source_cancellation_keeps_unconfirmed_record_for_restart(tmp_path: Path) -> None:
    import asyncio

    source, checkpoint = tmp_path / "input.jsonl", tmp_path / "checkpoint.json"
    source.write_text(json.dumps(PAYLOAD) + "\n")
    publisher = AsyncMock()
    publisher.publish.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await publish_file(source, checkpoint, publisher)
    assert load_checkpoint(source, checkpoint).accepted == 0
    first_id = publisher.publish.call_args.args[0].task_id
    publisher.publish.side_effect = None
    await publish_file(source, checkpoint, publisher)
    assert publisher.publish.call_args.args[0].task_id == first_id


def test_changed_input_cannot_reuse_checkpoint(tmp_path: Path) -> None:
    source, checkpoint = tmp_path / "input.jsonl", tmp_path / "checkpoint.json"
    source.write_text("one")
    load_checkpoint(source, checkpoint).save(checkpoint)
    source.write_text("two")
    with pytest.raises(ValueError, match="differs"):
        load_checkpoint(source, checkpoint)


async def test_oversized_record_is_not_loaded_or_published(tmp_path: Path) -> None:
    source = tmp_path / "input.jsonl"
    source.write_text("x" * 2_100_001)
    publisher = AsyncMock()
    with pytest.raises(ValueError, match="maximum"):
        await publish_file(source, tmp_path / "checkpoint.json", publisher)
    publisher.publish.assert_not_called()


def test_task_identity_ignores_delivery_timestamp_and_tracks_payload() -> None:
    import uuid

    key = str(uuid.uuid4())
    body = {"task_id": key, "created_at": "old", "payload": PAYLOAD}
    assert message_identity(body, key) == message_identity(body | {"created_at": "new"}, key)
    assert message_identity("bad", "not-a-uuid") == message_identity("bad", "not-a-uuid")
    assert TaskPayload(**PAYLOAD).prompt == PAYLOAD["prompt"]
