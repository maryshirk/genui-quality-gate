import re
import sys
from pathlib import Path

SUBJECT = re.compile(
    r"(?:feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)"
    r"(?:\([a-z0-9_-]+\))?!?: \S[^\r\n]*"
)


def valid_subject(subject: str) -> bool:
    return bool(SUBJECT.fullmatch(subject)) and len(subject) <= 100


def main() -> int:
    lines = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    if lines and valid_subject(lines[0]):
        return 0
    print("Use a Conventional Commit subject, up to 100 characters: feat: add health checks")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
