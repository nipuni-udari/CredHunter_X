# CredHunter-X

LLM-assisted secret detection and explanation for Python source code, with a
privacy-preserving masking layer that measures the cost of never sending real
secret values to a third-party LLM.

## Status

Project scaffold only — no functional code yet. Implementation proceeds
milestone by milestone.

## Setup

Requires [`uv`](https://github.com/astral-sh/uv) and
[`gitleaks`](https://github.com/gitleaks/gitleaks) on `PATH`.

```powershell
uv sync
uv run pre-commit install
```

## License

TBD — choose a license (e.g. MIT, Apache-2.0) before making this repository
public.
