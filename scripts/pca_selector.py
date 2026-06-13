import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class PCA:
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.strong_pairs_: pd.DataFrame | None = None
        self.corr_matrix_: pd.DataFrame | None = None

    @staticmethod
    def encode_non_numeric(data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        for col in df.select_dtypes(include=["object", "datetime64[ns]"]).columns:
            df[col] = df[col].astype("category").cat.codes
        return df

    @staticmethod
    def plot_heatmap(corr: pd.DataFrame) -> None:
        plt.figure(figsize=(40, 20))
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.75},
        )
        plt.title("Correlation Matrix (numeric columns)")
        plt.tight_layout()
        plt.show()

    def extract_strong_pairs(self, corr: pd.DataFrame) -> pd.DataFrame:
        upper = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        pairs = corr.where(upper).stack().reset_index()
        pairs.columns = ["feature_a", "feature_b", "corr"]
        strong = pairs.loc[pairs["corr"].abs() > self.threshold].copy()
        strong = strong.sort_values("corr", key=lambda s: s.abs(), ascending=False)
        self.strong_pairs_ = strong
        return strong

    def print_strong_pairs(self) -> None:
        if self.strong_pairs_ is None or self.strong_pairs_.empty:
            print(f"\nNo pairs with |correlation| > {self.threshold}")
            return
        print(f"\nPairs with |correlation| > {self.threshold}:")
        print(self.strong_pairs_.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    def analyze(self, data: pd.DataFrame) -> None:
        encoded = self.encode_non_numeric(data)
        corr = encoded.corr(numeric_only=True)
        self.corr_matrix_ = corr

        self.plot_heatmap(corr)
        self.extract_strong_pairs(corr)
        self.print_strong_pairs()

    def suggest_drop(self, data: pd.DataFrame) -> list[str]:
        if self.strong_pairs_ is None:
            self.analyze(data)
        drop: set[str] = set()
        for _, row in self.strong_pairs_.iterrows():
            drop.add(row["feature_b"])
        return list(drop)
