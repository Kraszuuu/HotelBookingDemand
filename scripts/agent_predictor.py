import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class AgentImputer:
    def __init__(self, random_state: int = 268555):
        self.random_state = random_state
        self.model: Any | None = None

    def _split_datasets(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train = data[data["agent"].notna()].copy()
        predict = data[data["agent"].isna()].copy()
        return train, predict

    def _build_feature_sets(
        self, train: pd.DataFrame, predict: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        y = train["agent"]
        X = train.drop(columns=["agent"])
        pred_X = predict.drop(columns=["agent"])
        return X, y, pred_X

    def _get_feature_columns(self, X: pd.DataFrame) -> tuple[list[str], list[str]]:
        cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
        num = X.select_dtypes(include=["number"]).columns.tolist()
        return cat, num

    def _build_pipeline(self, cat_cols: list[str], num_cols: list[str]) -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ("num", "passthrough", num_cols),
            ]
        )
        return Pipeline([
            ("pre", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=self.random_state))
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AgentImputer":
        train_X, val_X, train_y, val_y = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        self.model = self._build_pipeline(*self._get_feature_columns(X))
        self.model.fit(train_X, train_y)
        print("val score:", self.model.score(val_X, val_y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        return self.model.predict(X)

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        train_df, predict_df = self._split_datasets(result)
        if predict_df.empty:
            return result
        X, y, pred_X = self._build_feature_sets(train_df, predict_df)
        self.fit(X, y)
        predicted = self.predict(pred_X)
        result.loc[result["agent"].isna(), "agent"] = predicted
        return result


class AgentPredictor:
    def __init__(self, random_state: int = 268555):
        self.random_state = random_state

    def _load_default_data(self) -> pd.DataFrame:
        candidates = [Path("./Dataset/hotel_bookings.csv"), Path("./Data/hotel_bookings.csv")]
        for p in candidates:
            if p.exists():
                return pd.read_csv(p)
        raise FileNotFoundError("hotel_bookings.csv not found in ./Dataset or ./Data")

    def train_cancellation_model(
        self,
        data: pd.DataFrame | None = None,
        test_size: float = 0.20,
        validation_size: float = 0.20,
        classifier: str = "random_forest",
        classifier_params: dict[str, object] | None = None,
        tune_model: bool = True,
        tuning_metric: str = "f1",
        n_iter: int = 20,
        cv_folds: int = 3,
    ) -> dict[str, object]:
        df = data.copy() if data is not None else self._load_default_data()

        valid_statuses = ["Check-Out", "Canceled", "No-Show"]
        df = df.loc[df["reservation_status"].isin(valid_statuses)].copy()
        df["target_cancelled"] = np.where(
            df["reservation_status"].isin(["Canceled", "No-Show"]), 1, 0
        )

        drop_cols = ["target_cancelled", "reservation_status", "reservation_status_date", "is_canceled"]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        y = df["target_cancelled"].astype(int)

        for col in X.select_dtypes(include=["datetime64[ns]"]).columns:
            X[col] = X[col].map(lambda v: v.value if pd.notna(v) else np.nan)

        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]), cat_cols),
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median"))
                ]), num_cols)
            ]
        )

        clf_name = classifier.strip().lower()
        override = classifier_params or {}

        if clf_name == "random_forest":
            params = {"n_estimators": 300, "random_state": self.random_state, "n_jobs": -1, "class_weight": "balanced"}
            params.update(override)
            estimator = RandomForestClassifier(**params)
            param_dist = {
                "clf__n_estimators": [200, 300, 500, 800],
                "clf__max_depth": [None, 10, 20, 30, 40],
                "clf__min_samples_split": [2, 5, 10, 20],
                "clf__min_samples_leaf": [1, 2, 4, 8],
                "clf__max_features": ["sqrt", "log2", None],
                "clf__class_weight": ["balanced", "balanced_subsample", None]
            }
        elif clf_name == "extra_trees":
            params = {"n_estimators": 400, "random_state": self.random_state, "n_jobs": -1, "class_weight": "balanced"}
            params.update(override)
            estimator = ExtraTreesClassifier(**params)
            param_dist = {
                "clf__n_estimators": [200, 400, 700, 1000],
                "clf__max_depth": [None, 10, 20, 30, 40],
                "clf__min_samples_split": [2, 5, 10, 20],
                "clf__min_samples_leaf": [1, 2, 4, 8],
                "clf__max_features": ["sqrt", "log2", None],
                "clf__class_weight": ["balanced", "balanced_subsample", None]
            }
        elif clf_name == "logistic_regression":
            params = {"C": 1.0, "l1_ratio": 0, "solver": "liblinear", "class_weight": "balanced", "max_iter": 2000}
            params.update(override)
            params.setdefault("random_state", self.random_state)
            estimator = LogisticRegression(**params)
            param_dist = {
                "clf__C": [0.01, 0.1, 1.0, 3.0, 10.0],
                "clf__solver": ["liblinear", "lbfgs"],
                "clf__class_weight": ["balanced", None],
                "clf__max_iter": [1000, 2000, 4000]
            }
        else:
            raise ValueError("Unsupported classifier. Use: 'random_forest', 'extra_trees', 'logistic_regression'.")

        model = Pipeline([("pre", preprocessor), ("clf", estimator)])

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        val_rel_size = validation_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_rel_size, random_state=self.random_state, stratify=y_train_val
        )

        tuning_summary: dict[str, object] | None = None
        if tune_model:
            search = RandomizedSearchCV(
                estimator=model, param_distributions=param_dist, n_iter=n_iter,
                scoring=tuning_metric, cv=cv_folds, n_jobs=-1,
                random_state=self.random_state, verbose=1
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            tuning_summary = {
                "classifier": clf_name, "bestParams": search.best_params_,
                "bestCvScore": float(search.best_score_), "metric": tuning_metric,
                "cvFolds": cv_folds, "nIter": n_iter
            }
            print(f"\nTuning summary:\nClassifier: {clf_name}\nBest CV {tuning_metric}: {tuning_summary['bestCvScore']:.4f}\nBest params: {tuning_summary['bestParams']}")
        else:
            model.fit(X_train, y_train)

        def build_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, object]:
            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "confusion_matrix": confusion_matrix(y_true, y_pred)
            }

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        val_metrics = build_metrics(y_val, val_pred)
        test_metrics = build_metrics(y_test, test_pred)

        print(f"Train: {X_train.shape[0]} ({X_train.shape[0] / len(X):.2%})")
        print(f"Validation: {X_val.shape[0]} ({X_val.shape[0] / len(X):.2%})")
        print(f"Test: {X_test.shape[0]} ({X_test.shape[0] / len(X):.2%})")

        for label, metrics in [("Validation", val_metrics), ("Test", test_metrics)]:
            print(f"\n{label} metrics:")
            for k in ["accuracy", "precision", "recall", "f1"]:
                print(f"{k.capitalize()}:  {metrics[k]:.4f}")
            cm = metrics["confusion_matrix"]
            print(pd.DataFrame(cm, index=["Actual_NotCancelled", "Actual_Cancelled"],
                               columns=["Pred_NotCancelled", "Pred_Cancelled"]))

        return {
            "model": model, "classifier": clf_name, "classifierParams": params,
            "XTrain": X_train, "XVal": X_val, "XTest": X_test,
            "yTrain": y_train, "yVal": y_val, "yTest": y_test,
            "validationMetrics": val_metrics, "testMetrics": test_metrics,
            "tuningSummary": tuning_summary
        }

    def run_variant_sweep(
        self,
        base_data: pd.DataFrame,
        modified_data: pd.DataFrame | None = None,
        output_path: str = "./CSVs/cancellation_classifier_variant_results.csv",
        base_data_type: str = "original",
    ) -> pd.DataFrame:
        frames: dict[str, pd.DataFrame] = {base_data_type: base_data}
        if modified_data is not None:
            frames["modified"] = modified_data

        variants: dict[str, list[dict[str, object]]] = {
            "random_forest": [
                {"variant": "baseline", "params": {"n_estimators": 300, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt", "class_weight": "balanced"}},
                {"variant": "regularized", "params": {"n_estimators": 250, "max_depth": 12, "min_samples_split": 10, "min_samples_leaf": 2, "max_features": "sqrt", "class_weight": "balanced"}},
                {"variant": "wider", "params": {"n_estimators": 500, "max_depth": 20, "min_samples_split": 5, "min_samples_leaf": 1, "max_features": "log2", "class_weight": "balanced_subsample"}},
            ],
            "extra_trees": [
                {"variant": "baseline", "params": {"n_estimators": 400, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt", "class_weight": "balanced"}},
                {"variant": "regularized", "params": {"n_estimators": 300, "max_depth": 12, "min_samples_split": 10, "min_samples_leaf": 2, "max_features": "sqrt", "class_weight": "balanced"}},
                {"variant": "wider", "params": {"n_estimators": 700, "max_depth": 20, "min_samples_split": 5, "min_samples_leaf": 1, "max_features": "log2", "class_weight": "balanced_subsample"}},
            ],
            "logistic_regression": [
                {"variant": "baseline", "params": {"C": 1.0, "solver": "liblinear", "class_weight": "balanced", "max_iter": 2000}},
                {"variant": "strong_regularization", "params": {"C": 0.1, "solver": "liblinear", "class_weight": "balanced", "max_iter": 3000}},
                {"variant": "lighter_regularization", "params": {"C": 3.0, "solver": "lbfgs", "class_weight": None, "max_iter": 4000}},
            ],
        }

        def param_cols(p: dict[str, object]) -> dict[str, object]:
            return {k: p.get(k) for k in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "class_weight", "C", "solver", "max_iter"]}

        rows: list[dict[str, object]] = []
        for df_type, frame in frames.items():
            for clf_name, clf_variants in variants.items():
                for idx, v in enumerate(clf_variants, start=1):
                    print(f"\nRunning {df_type} / {clf_name} / {v['variant']}")
                    result = self.train_cancellation_model(
                        frame, classifier=clf_name, classifier_params=v["params"],
                        tune_model=False
                    )
                    row = {
                        "dataFrameType": df_type, "classifier": clf_name,
                        "variant": v["variant"], "variantIndex": idx,
                        "Accuracy": result["testMetrics"]["accuracy"],
                        "Precision": result["testMetrics"]["precision"],
                        "Recall": result["testMetrics"]["recall"],
                        "F1": result["testMetrics"]["f1"],
                    }
                    row.update(param_cols(result["classifierParams"]))
                    rows.append(row)

        results_df = pd.DataFrame(rows)
        results_df = results_df.sort_values(["dataFrameType", "classifier", "variantIndex"]).reset_index(drop=True)

        numeric_cols = results_df.select_dtypes(include=["number"]).columns.tolist()
        round_cols = [c for c in numeric_cols if c != "variantIndex"]
        results_df[round_cols] = results_df[round_cols].round(3)

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        exists = out_path.exists()
        results_df.to_csv(out_path, index=False, encoding="utf-8-sig", mode="a", header=not exists)
        print(f"\nAppended results to {output_path}")
        print(results_df.to_string(index=False))

        return results_df
