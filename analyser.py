from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


class Analyser:
    def train_adr_model(
        self,
        data: pd.DataFrame | None = None,
        test_size: float = 0.20,
        validation_size: float = 0.20,
        random_state: int = 268555,
        regressor: str = "random_forest",
        regressor_params: dict[str, Any] | None = None,
        tune: bool = True,
        n_iter: int = 20,
        cv_folds: int = 3,
    ) -> dict[str, Any]:
        if data is None:
            paths = [Path("./Dataset/hotel_bookings.csv"), Path("./Data/hotel_bookings.csv")]
            existing = [p for p in paths if p.exists()]
            if not existing:
                raise FileNotFoundError("hotel_bookings.csv not found.")
            df = pd.read_csv(existing[0])
        else:
            df = data.copy()

        valid = ["Check-Out", "Canceled", "No-Show"]
        df = df.loc[df["reservation_status"].isin(valid)].copy()
        df = df.loc[df["adr"].notna()].copy()

        upper = df["adr"].quantile(0.99)
        df = df.loc[df["adr"] <= upper].copy()

        drop_cols = ["adr", "reservation_status", "reservation_status_date", "is_canceled"]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        y = df["adr"].astype(float)

        for col in X.select_dtypes(include=["datetime64[ns]"]).columns:
            X[col] = X[col].map(lambda v: v.value if pd.notna(v) else float("nan"))

        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()

        pre = ColumnTransformer(transformers=[
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ])

        sel = regressor.strip().lower()
        overrides = regressor_params or {}
        if sel == "random_forest":
            params = {"n_estimators": 300, "random_state": random_state, "n_jobs": -1}
            params.update(overrides)
            est = RandomForestRegressor(**params)
            dist = {
                "reg__n_estimators": [200, 300, 500, 800],
                "reg__max_depth": [None, 10, 20, 30, 40],
                "reg__min_samples_split": [2, 5, 10, 20],
                "reg__min_samples_leaf": [1, 2, 4, 8],
                "reg__max_features": ["sqrt", "log2", None],
            }
        elif sel == "extra_trees":
            params = {"n_estimators": 400, "random_state": random_state, "n_jobs": -1}
            params.update(overrides)
            est = ExtraTreesRegressor(**params)
            dist = {
                "reg__n_estimators": [200, 400, 700, 1000],
                "reg__max_depth": [None, 10, 20, 30, 40],
                "reg__min_samples_split": [2, 5, 10, 20],
                "reg__min_samples_leaf": [1, 2, 4, 8],
                "reg__max_features": ["sqrt", "log2", None],
            }
        elif sel == "ridge":
            params = {"alpha": 1.0, "random_state": random_state}
            params.update(overrides)
            est = Ridge(**params)
            dist = {"reg__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
        else:
            raise ValueError("Unsupported regressor. Use 'random_forest', 'extra_trees', 'ridge'.")

        model = Pipeline([("pre", pre), ("reg", est)])

        X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        val_ratio = validation_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=val_ratio, random_state=random_state)

        tune_summary = None
        if tune:
            search = RandomizedSearchCV(
                estimator=model, param_distributions=dist, n_iter=n_iter,
                scoring="neg_mean_squared_error", cv=cv_folds,
                n_jobs=-1, random_state=random_state, verbose=1,
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            tune_summary = {
                "regressor": sel, "bestParams": search.best_params_,
                "bestCvScore": float(search.best_score_),
                "metric": "neg_mean_squared_error", "cvFolds": cv_folds, "nIter": n_iter,
            }
            print(f"\nTuning summary:\nRegressor: {sel}")
            print(f"Best CV neg MSE: {tune_summary['bestCvScore']:.4f}")
            print("Best params:", tune_summary["bestParams"])
        else:
            model.fit(X_train, y_train)

        def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
            return {
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "r2": float(r2_score(y_true, y_pred)),
                "mse": float(mean_squared_error(y_true, y_pred)),
            }

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        val_metrics = _metrics(y_val, val_pred)
        test_metrics = _metrics(y_test, test_pred)

        print(f"Train: {X_train.shape[0]} ({X_train.shape[0] / len(X):.2%})")
        print(f"Validation: {X_val.shape[0]} ({X_val.shape[0] / len(X):.2%})")
        print(f"Test: {X_test.shape[0]} ({X_test.shape[0] / len(X):.2%})")
        print(f"\nValidation metrics:\nRMSE: {val_metrics['rmse']:.4f}\nMAE:  {val_metrics['mae']:.4f}")
        print(f"R2:   {val_metrics['r2']:.4f}\nMSE:  {val_metrics['mse']:.4f}")
        print(f"\nTest metrics:\nRMSE: {test_metrics['rmse']:.4f}\nMAE:  {test_metrics['mae']:.4f}")
        print(f"R2:   {test_metrics['r2']:.4f}\nMSE:  {test_metrics['mse']:.4f}")

        return {
            "model": model, "regressor": sel, "regressorParams": params,
            "XTrain": X_train, "XVal": X_val, "XTest": X_test,
            "yTrain": y_train, "yVal": y_val, "yTest": y_test,
            "validationMetrics": val_metrics, "testMetrics": test_metrics,
            "tuningSummary": tune_summary,
        }

    def plot_adr_histogram(self, data: pd.DataFrame, quantiles: tuple[float, float] = (0, 1)) -> None:
        s = data["adr"].dropna()
        lo, hi = float(s.quantile(quantiles[0])), float(s.quantile(quantiles[1]))
        ps = s[(s >= lo) & (s <= hi)]
        fig, ax = plt.subplots(figsize=(20, 5))
        ax.hist(ps, bins=400, color="#2e8b57", edgecolor="#000000", alpha=0.8)
        ax.set_title(f"ADR histogram ({quantiles[0]} to {quantiles[1]} percentiles)")
        ax.set_xlabel("ADR")
        ax.set_ylabel("Number of bookings")
        x_min, x_max = int(np.floor(ps.min() / 50) * 50), int(np.ceil(ps.max() / 50) * 50)
        if x_min == x_max:
            x_max = x_min + 50
        ax.set_xlim(x_min, x_max)
        ax.set_xticks(np.arange(x_min, x_max + 1, 25))
        fig.tight_layout()
        plt.show()

    def print_room_match_summary(self, data: pd.DataFrame) -> None:
        def _pct(df: pd.DataFrame) -> pd.DataFrame:
            return (df.assign(
                room_match=np.where(df["assigned_room_type"] == df["reserved_room_type"],
                                    "Same room type", "Different room type")
            )["room_match"].value_counts(dropna=False, normalize=True).mul(100)
             .round(2).rename("percentage").reset_index()
             .rename(columns={"index": "room_match"}))

        for label, subset in [("ADR <= 0", data.loc[data["adr"] <= 0]),
                               ("ADR > 0", data.loc[data["adr"] > 0]),
                               ("All records", data)]:
            print(f"\n{label}:")
            print(_pct(subset).to_string(index=False))

    def print_mismatch_impact(self, data: pd.DataFrame) -> None:
        cols = ["hotel", "adr", "assigned_room_type", "reserved_room_type"]
        missing = [c for c in cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        w = data[cols].dropna(subset=cols).copy()
        rows: list[dict[str, Any]] = []
        for hotel in sorted(w["hotel"].unique()):
            hd = w[w["hotel"] == hotel]
            total = float(hd["adr"].sum())
            md = hd[hd["assigned_room_type"] != hd["reserved_room_type"]]
            mismatch_sum = float(md["adr"].sum())
            med = hd.groupby("reserved_room_type")["adr"].median()
            weighted = float(md["reserved_room_type"].map(med).fillna(0.0).sum())
            diff = mismatch_sum - weighted
            rows.append({
                "hotel": hotel, "total_adr": round(total, 2),
                "mismatch_adr_sum": round(mismatch_sum, 2),
                "mismatch_count_x_reserved_type_median_adr": round(weighted, 2),
                "absolute_difference": round(diff, 2),
                "difference_to_total_ratio": round(diff / total, 6) if total else float("nan"),
            })
        result = pd.DataFrame(rows)
        if result.empty:
            print("No data after filtering.")
            return
        print("ADR mismatch impact summary by hotel:")
        print(result.to_string(index=False))

    def print_room_type_summary(self, data: pd.DataFrame) -> None:
        base = data.copy()
        base["arrival_date"] = pd.to_datetime(base["arrival_date"], errors="coerce")
        base["total_nights"] = (base["stays_in_week_nights"].fillna(0) + base["stays_in_weekend_nights"].fillna(0)).astype(int)
        w = base[["hotel", "assigned_room_type", "arrival_date", "total_nights", "adr"]].dropna(
            subset=["hotel", "assigned_room_type", "arrival_date", "adr"])
        w = w[w["total_nights"] > 0].copy()

        adr_by = w.groupby(["hotel", "assigned_room_type"], as_index=False)["adr"].agg(
            adr_mean="mean", adr_median="median", adr_min="min", adr_max="max")

        expanded = w.loc[w.index.repeat(w["total_nights"])].copy()
        expanded["day_offset"] = expanded.groupby(level=0).cumcount()
        expanded["stay_date"] = expanded["arrival_date"] + pd.to_timedelta(expanded["day_offset"], unit="D")
        daily = expanded.groupby(["hotel", "assigned_room_type", "stay_date"]).size().rename("occupied_rooms").reset_index()
        max_occ = daily.groupby(["hotel", "assigned_room_type"], as_index=False)["occupied_rooms"].max().rename(
            columns={"occupied_rooms": "max_occupied_rooms"})

        summary = adr_by.merge(max_occ, on=["hotel", "assigned_room_type"], how="left").sort_values(
            ["hotel", "assigned_room_type"])
        summary["adr_mean"] = summary["adr_mean"].round(2)

        for hotel in sorted(summary["hotel"].unique()):
            print(f"\n{hotel}:")
            hd = summary[summary["hotel"] == hotel][
                ["assigned_room_type", "adr_mean", "adr_median", "adr_min", "adr_max", "max_occupied_rooms"]
            ]
            print(hd.to_string(index=False))
