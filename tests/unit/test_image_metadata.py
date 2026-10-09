import pytest

from scripts.image_metadata import image_metadata

SHA = "a" * 40
IMAGE = "ghcr.io/maryshirk/genui-quality-gate"


def test_feature_branch_has_commit_specific_tags_only() -> None:
    result = image_metadata("0.1.0", SHA, "Maryshirk/GenUI-Quality-Gate", "refs/heads/feat/x")
    assert result.tags == (f"{IMAGE}:sha-{SHA}", f"{IMAGE}:0.1.0-sha{'a' * 12}")


def test_main_has_a_moving_main_tag_but_no_latest() -> None:
    result = image_metadata("0.1.0", SHA, "maryshirk/genui-quality-gate", "refs/heads/main")
    assert result.tags[-1] == f"{IMAGE}:main"
    assert f"{IMAGE}:latest" not in result.tags


def test_release_matches_the_package_version() -> None:
    result = image_metadata("1.2.3", SHA, "maryshirk/genui-quality-gate", "refs/tags/v1.2.3")
    assert result.tags[-2:] == (f"{IMAGE}:1.2.3", f"{IMAGE}:latest")


@pytest.mark.parametrize("tag", ["v0.2.0", "v0.1.0-rc1", "v01.1.0", "v0.1.0\nlatest"])
def test_misleading_release_tags_are_rejected(tag: str) -> None:
    with pytest.raises(ValueError, match="Release tag"):
        image_metadata("0.1.0", SHA, "maryshirk/genui-quality-gate", f"refs/tags/{tag}")


@pytest.mark.parametrize("version", ["0.1", "v0.1.0", "01.1.0", "0.1.0-rc1", "0.1.0\n"])
def test_invalid_versions_are_rejected(version: str) -> None:
    with pytest.raises(ValueError, match="Package version"):
        image_metadata(version, SHA, "maryshirk/genui-quality-gate", "refs/heads/main")


@pytest.mark.parametrize("sha", ["abc123", "G" * 40, "$(touch unsafe)"])
def test_invalid_revision_is_rejected(sha: str) -> None:
    with pytest.raises(ValueError, match="GITHUB_SHA"):
        image_metadata("0.1.0", sha, "maryshirk/genui-quality-gate", "refs/heads/main")


@pytest.mark.parametrize("repository", ["repo", "owner/repo/extra", "owner/repo\nunsafe"])
def test_invalid_repository_is_rejected(repository: str) -> None:
    with pytest.raises(ValueError, match="GITHUB_REPOSITORY"):
        image_metadata("0.1.0", SHA, repository, "refs/heads/main")
