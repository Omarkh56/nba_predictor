"""Load project-root .env and require API keys with a clear error."""

import os
import sys

from dotenv import load_dotenv

load_dotenv()


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(
            f"{name} not set — copy .env.example to .env and fill in your key",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return value
