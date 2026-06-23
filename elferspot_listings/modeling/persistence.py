from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SkopsNotInstalledError(ImportError):
    pass


def _skops_io():
    try:
        import skops.io as sio
    except ModuleNotFoundError as exc:
        raise SkopsNotInstalledError("skops is not installed") from exc

    return sio


def save_sklearn_model(model: Any, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _skops_io().dump(model, output_path)
    return output_path


def inspect_sklearn_model(path: str | Path) -> list[str]:
    return list(_skops_io().get_untrusted_types(file=Path(path)))


def load_sklearn_model(path: str | Path, trusted: list[str]):
    return _skops_io().load(Path(path), trusted=trusted)


def write_model_card(path: str | Path, metadata: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_name = str(metadata.get("model_name", "unknown_model"))
    purpose = str(
        metadata.get(
            "purpose",
            "Predict Porsche listing prices from cleaned Gold-layer listing data.",
        )
    )
    target = str(metadata.get("target", "price_in_eur"))
    metrics = metadata.get("metrics", {})
    limitations = list(metadata.get("limitations", []))
    usage_notes = list(metadata.get("usage_notes", []))

    limitation_lines = "\n".join(f"- {item}" for item in limitations) if limitations else "- Not specified"
    usage_lines = (
        "\n".join(f"- {item}" for item in usage_notes)
        if usage_notes
        else "- Load `.skops` artifacts with `skops.io.load` after inspecting untrusted types with `skops.io.get_untrusted_types`."
    )

    content = f"""# Model Card: {model_name}

## Purpose

{purpose}

## Target

`{target}`

## Metrics

```json
{json.dumps(metrics, indent=2, sort_keys=True)}
```

## Limitations

{limitation_lines}

## Usage Notes

{usage_lines}
"""

    output_path.write_text(content, encoding="utf-8")
    return output_path
