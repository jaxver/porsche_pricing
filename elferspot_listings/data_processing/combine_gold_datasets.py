from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import config


DEFAULT_OUTPUT = Path(config.LISTINGS_GOLD_COMBINED)
COMPLETENESS_BONUS = {
    "URL": 8,
    "price_in_eur": 5,
    "Mileage_km": 3,
    "Description": 2,
    "Scraped_At": 1,
}


def _normalize_gold_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "URL" not in df.columns:
        raise ValueError("Gold dataset is missing required URL column")
    normalized = df.copy()
    normalized["URL"] = normalized["URL"].astype("string").str.strip()
    normalized = normalized.loc[normalized["URL"].notna() & (normalized["URL"] != "")].copy()
    return normalized  # type: ignore[return-value]


def score_row_completeness(df: pd.DataFrame) -> pd.Series:
    score = df.notna().sum(axis=1).astype(float)
    for column, bonus in COMPLETENESS_BONUS.items():
        if column in df.columns:
            score = score + df[column].notna().astype(float) * bonus
    return score


def combine_gold_datasets(old_path: str | Path, new_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    old_frame = _normalize_gold_frame(pd.read_excel(old_path))
    new_frame = _normalize_gold_frame(pd.read_excel(new_path))

    combined = pd.concat([old_frame, new_frame], ignore_index=True, sort=False)
    combined["_completeness_score"] = score_row_completeness(combined)
    combined["_scraped_sort"] = pd.to_datetime(combined["Scraped_At"], errors="coerce") if "Scraped_At" in combined.columns else pd.NaT
    combined = combined.sort_values(
        by=["URL", "_completeness_score", "_scraped_sort"],
        ascending=[True, False, False],
        kind="mergesort",
    )
    combined = combined.drop_duplicates(subset=["URL"], keep="first").drop(columns=["_completeness_score", "_scraped_sort"])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_excel(output_path, index=False)
    return combined


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combine two gold datasets into a single deduplicated gold file.")
    parser.add_argument("--old", type=Path, required=True, help="Path to the older clean gold file.")
    parser.add_argument("--new", type=Path, required=True, help="Path to the newer clean gold file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path for the combined gold file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    combined = combine_gold_datasets(args.old, args.new, args.output)
    print(f"Wrote combined gold file to {args.output}")
    print(f"Rows kept after URL dedupe: {len(combined)}")
    print(
        "Use: "
        f'& "C:\\Users\\jaxon\\.venvs\\Elferspot_prod\\Scripts\\python.exe" -m elferspot_listings.modeling.cli --input "{args.output}" --model all --include-optionals --verbose'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
