import pytest

from scripts.check_commit_message import valid_subject


@pytest.mark.parametrize(
    "subject", ["feat: add health endpoints", "fix(api): handle timeout", "feat!: change schema"]
)
def test_accepts_conventional_subject(subject: str) -> None:
    assert valid_subject(subject)


@pytest.mark.parametrize("subject", ["", "update stuff", "feat:", "fix: ", "feat: " + "x" * 100])
def test_rejects_invalid_subject(subject: str) -> None:
    assert not valid_subject(subject)
