"""Allow running `python3 -m codex_2_agy ...` directly."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
