from pathlib import Path
import pandas as pd


class PredictionAndEvaluationV1:
    def __init__(self, output_dir: str = "./CSVs"):
        self.output_dir = Path(output_dir)

    def print_value_counts(self, data: pd.DataFrame) -> None:
        print(data.nunique(dropna=False))

        rows = []
        for col in data.columns:
            counts = data[col].value_counts(dropna=False).sort_values(ascending=False)
            rows.extend(
                {"column": col, "value": str(val), "count": int(cnt)}
                for val, cnt in counts.items()
            )

        result = pd.DataFrame(rows)
        out_path = self.output_dir / "column_value_counts.csv"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Wrote {len(result)} rows to {out_path}")
