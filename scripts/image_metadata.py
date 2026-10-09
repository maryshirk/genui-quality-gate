import argparse
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

SEMVER = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")


@dataclass(frozen=True)
class ImageMetadata:
    version: str
    image: str
    tags: tuple[str, ...]


def image_metadata(version: str, sha: str, repository: str, ref: str) -> ImageMetadata:
    if not SEMVER.fullmatch(version):
        raise ValueError("Package version must be a stable MAJOR.MINOR.PATCH version")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("GITHUB_SHA must be a full lowercase Git commit SHA")
    repository = repository.lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*/[a-z0-9][a-z0-9_.-]*", repository):
        raise ValueError("GITHUB_REPOSITORY must have the form owner/repository")

    image = f"ghcr.io/{repository}"
    tags = [f"{image}:sha-{sha}", f"{image}:{version}-sha{sha[:12]}"]
    if ref.startswith("refs/tags/"):
        if ref != f"refs/tags/v{version}":
            raise ValueError("Release tag must exactly match v<version> in pyproject.toml")
        tags.extend([f"{image}:{version}", f"{image}:latest"])
    elif ref == "refs/heads/main":
        tags.append(f"{image}:main")
    return ImageMetadata(version=version, image=image, tags=tuple(tags))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    with Path("pyproject.toml").open("rb") as file:
        version = tomllib.load(file)["project"]["version"]
    metadata = image_metadata(
        version, os.environ["GITHUB_SHA"], os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_REF"]
    )
    result = (
        f"version={metadata.version}\nimage={metadata.image}\n"
        f"primary_tag={metadata.tags[0]}\ntags<<IMAGE_TAGS\n"
        + "\n".join(metadata.tags)
        + "\nIMAGE_TAGS\n"
    )
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as file:
            file.write(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
