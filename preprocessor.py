import pandas as pd
import numpy as np
from pathlib import Path
from agent_predictor import AgentImputer


class Preprocessor:
    def __init__(self, random_state: int = 268555):
        self.random_state = random_state

    def load(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    @staticmethod
    def _get_month_map() -> dict[str, int]:
        return {
            "January": 1, "February": 2, "March": 3, "April": 4,
            "May": 5, "June": 6, "July": 7, "August": 8,
            "September": 9, "October": 10, "November": 11, "December": 12,
        }

    @staticmethod
    def _build_arrival_date(data: pd.DataFrame) -> pd.Series:
        return pd.to_datetime(
            {
                "year": data["arrival_date_year"],
                "month": data["arrival_date_month"],
                "day": data["arrival_date_day_of_month"],
            },
            errors="coerce",
        )

    @staticmethod
    def _check_time_consistency(data: pd.DataFrame) -> pd.DataFrame:
        checkout = data.loc[data["reservation_status"] == "Check-Out"].copy()
        days_diff = (checkout["reservation_status_date"] - checkout["arrival_date"]).dt.days
        expected = checkout["stays_in_week_nights"] + checkout["stays_in_weekend_nights"]
        invalid = checkout.index[days_diff != expected]
        if len(invalid) > 0:
            print(f"Dropped invalid rows: {len(invalid)}")
        return data.drop(index=invalid).copy()

    def _prepare_time(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"], errors="coerce")
        df["arrival_date_month"] = df["arrival_date_month"].map(self._get_month_map())
        df["arrival_date"] = self._build_arrival_date(df)
        df = self._check_time_consistency(df)
        df = df[df["arrival_date"].notna()].copy()
        df.drop(columns=["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"], inplace=True)
        return df

    @staticmethod
    def print_missing_summary(data: pd.DataFrame) -> None:
        missing_count = data.isna().sum()
        missing_pct = (missing_count / len(data) * 100).round(2)
        summary = pd.DataFrame({"missingCount": missing_count, "missingPercentage": missing_pct})
        print("\nMissing data summary:")
        print(summary)

    def process(
        self,
        path: str,
        impute_agent: bool = True,
        print_info: bool = True,
    ) -> pd.DataFrame:
        df = self.load(path)

        if print_info:
            df.info()
            self.print_missing_summary(df)

        df.drop(columns=["company"], inplace=True)
        df = df.loc[df["country"].notna()].copy()
        df = df.loc[df["children"].notna()].copy()

        df = self._prepare_time(df)
        df = df[df["adults"] != 0].copy()

        if impute_agent:
            imputer = AgentImputer(random_state=self.random_state)
            df = imputer.fit_transform(df)

        print("\nAfter preprocessing:")
        self.print_missing_summary(df)

        return df
