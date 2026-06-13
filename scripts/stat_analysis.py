import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Any


class Statistics:
    def __init__(self):
        self.rng = np.random.default_rng(42)

    def mann_whitney_test(self, city_adr: pd.Series, resort_adr: pd.Series) -> dict[str, Any]:
        median_city = city_adr.median()
        median_resort = resort_adr.median()
        u_stat, p_value = stats.mannwhitneyu(city_adr, resort_adr, alternative="two-sided")
        delta = (2.0 * u_stat) / (len(city_adr) * len(resort_adr)) - 1.0

        print(f"Median ADR - City Hotel: {median_city:.2f}")
        print(f"Median ADR - Resort Hotel: {median_resort:.2f}")
        print(f"Mann-Whitney U statistic: {u_stat:.4f}")
        print(f"p-value: {p_value:.6g}")
        print(f"Cliff's Delta: {delta:.4f}")
        return {"u_stat": u_stat, "p_value": p_value, "cliffs_delta": delta,
                "median_city": median_city, "median_resort": median_resort}

    def bootstrap_cliffs_delta(self, city_adr: pd.Series, resort_adr: pd.Series,
                               n_samples: int = 5000, ci_level: float = 0.95) -> dict[str, Any]:
        alpha = 1.0 - ci_level
        deltas = []
        for _ in range(n_samples):
            c = self.rng.choice(city_adr, size=len(city_adr), replace=True)
            r = self.rng.choice(resort_adr, size=len(resort_adr), replace=True)
            u, _ = stats.mannwhitneyu(c, r, alternative="two-sided")
            deltas.append((2.0 * u) / (len(c) * len(r)) - 1.0)
        deltas_a = np.asarray(deltas)
        lo, hi = np.quantile(deltas_a, [alpha / 2.0, 1.0 - alpha / 2.0])
        observed = (2.0 * stats.mannwhitneyu(city_adr, resort_adr, alternative="two-sided")[0]) / (len(city_adr) * len(resort_adr)) - 1.0

        print(f"Cliff's Delta: {observed:.4f}")
        print(f"{ci_level * 100:.0f}% CI for Cliff's Delta: [{lo:.4f}, {hi:.4f}]")
        print("Interpretation: if the CI excludes 0, the effect is unlikely to be null.")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(deltas_a, bins=40, edgecolor="#1F1F1F", color="#1f7a1f", alpha=0.8)
        ax.axvline(observed, color="#b22222", linestyle="--", linewidth=2, label="Observed Delta")
        ax.axvline(lo, color="#f68115", linestyle=":", linewidth=2, label=f"{ci_level * 100:.0f}% CI")
        ax.axvline(hi, color="#f68115", linestyle=":", linewidth=2)
        ax.set_title("Bootstrap Distribution of Cliff's Delta")
        ax.set_xlabel("Cliff's Delta")
        ax.set_ylabel("Frequency")
        ax.legend()
        fig.tight_layout()
        plt.show()

        return {"observed": observed, "ci_low": lo, "ci_high": hi, "ci_level": ci_level,
                "boot_deltas": deltas_a}

    def cancellation_table(self, data: pd.DataFrame) -> pd.DataFrame:
        lo = data["adr"].quantile(0.01)
        hi = data["adr"].quantile(0.99)
        df = data[(data["adr"] >= lo) & (data["adr"] <= hi)].copy()
        df["quartile"] = pd.qcut(df["lead_time"], q=4, duplicates="drop")
        result = df.groupby("quartile", observed=False)["reservation_status"].apply(
            lambda s: s.isin(["Canceled", "No-Show"]).mean()
        ).reset_index(name="cancellation_rate")
        print("Cancellation rate by lead_time quartiles:")
        print(result)
        return result
