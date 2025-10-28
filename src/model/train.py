from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from catboost import CatBoostClassifier

from src.model.data_processing import DataFrameTransformer

DEFAULT_TARGET_COLUMN = "reservation_status"
DEFAULT_DATA_PATH = Path("src/data/train_data.csv")
DEFAULT_MODEL_PATH = Path("models/booking_status_pipeline.joblib")
DEFAULT_METRICS_PATH = Path("artifacts/training_metrics.json")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _normalize_label(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _drop_adr_outliers(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    upper_bound: float,
) -> tuple[pd.DataFrame, pd.Series, int]:
    """Drop rows where ADR exceeds the configured upper bound."""
    if "adr" not in X.columns:
        return X, y, 0

    mask = X["adr"].le(upper_bound)
    removed_count = int((~mask).sum())
    if removed_count == 0:
        return X, y, 0

    X_filtered = X.loc[mask].copy()
    y_filtered = y.loc[mask].copy()
    return X_filtered, y_filtered, removed_count


def _select_categorical_columns(df: pd.DataFrame) -> List[str]:
    return [
        col
        for col in df.columns
        if pd.api.types.is_object_dtype(df[col])
        or isinstance(df[col].dtype, CategoricalDtype)
    ]


class CatBoostPipelineClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper to auto-detect categorical columns when used inside a Pipeline."""

    def __init__(
        self,
        *,
        iterations: int = 1000,
        learning_rate: float = 0.05,
        depth: int = 8,
        l2_leaf_reg: float = 3.0,
        random_seed: int = 42,
        auto_class_weights: str | None = "Balanced",
        allow_writing_files: bool = False,
        verbose: int = 0,
    ) -> None:
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed
        self.auto_class_weights = auto_class_weights
        self.allow_writing_files = allow_writing_files
        self.verbose = verbose

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "l2_leaf_reg": self.l2_leaf_reg,
            "random_seed": self.random_seed,
            "auto_class_weights": self.auto_class_weights,
            "allow_writing_files": self.allow_writing_files,
            "verbose": self.verbose,
        }

    def set_params(self, **params: Any) -> "CatBoostPipelineClassifier":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> "CatBoostPipelineClassifier":
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        cat_features = _select_categorical_columns(X)

        self.model_ = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="TotalF1",
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_seed,
            auto_class_weights=self.auto_class_weights,
            allow_writing_files=self.allow_writing_files,
            verbose=self.verbose,
        )
        self.model_.fit(
            X,
            y,
            cat_features=cat_features,
            sample_weight=sample_weight,
        )
        self.feature_columns_ = list(X.columns)
        return self

    def _ensure_dataframe(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=self.feature_columns_)

    def predict(self, X: Any) -> np.ndarray:
        X_df = self._ensure_dataframe(X)
        preds = self.model_.predict(X_df)
        preds = np.asarray(preds).reshape(-1)
        return preds.astype(int)

    def predict_proba(self, X: Any) -> np.ndarray:
        X_df = self._ensure_dataframe(X)
        return self.model_.predict_proba(X_df)


def build_pipeline(
    *,
    random_state: int,
    model_type: str,
) -> Pipeline:
    """Assemble the end-to-end training pipeline."""
    feature_engineering = DataFrameTransformer(
        drop_room_type_columns=True,
        drop_adr_outliers=True,
        include_week_number_8to10_flag=True,
    )

    if model_type in {"histgb", "xgboost"}:
        numeric_processor = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_processor = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )

        preprocessing = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_processor,
                    make_column_selector(dtype_include=np.number),
                ),
                (
                    "categorical",
                    categorical_processor,
                    make_column_selector(dtype_include=object),
                ),
            ],
            remainder="drop",
        )

        if model_type == "histgb":
            classifier = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=400,
                max_leaf_nodes=31,
                min_samples_leaf=25,
                l2_regularization=1.0,
                random_state=random_state,
            )
        else:
            classifier = XGBClassifier(
                n_estimators=600,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                min_child_weight=3,
                objective="multi:softprob",
                eval_metric="mlogloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=random_state,
                verbosity=0,
            )

        return Pipeline(
            steps=[
                ("features", feature_engineering),
                ("preprocess", preprocessing),
                ("classifier", classifier),
            ]
        )

    if model_type == "catboost":
        classifier = CatBoostPipelineClassifier(
            iterations=1200,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3.0,
            random_seed=random_state,
            auto_class_weights=None,
            allow_writing_files=False,
            verbose=0,
        )
        return Pipeline(
            steps=[
                ("features", feature_engineering),
                ("classifier", classifier),
            ]
        )

    raise ValueError(
        f"Unsupported model type '{model_type}'. Choose 'histgb', 'catboost', or 'xgboost'."
    )


def get_param_distributions(model_type: str) -> Dict[str, Sequence[Any]]:
    if model_type == "histgb":
        return {
            "classifier__learning_rate": [0.03, 0.1],
            "classifier__max_iter": [250, 550],
            "classifier__max_leaf_nodes": [31, 127],
            "classifier__min_samples_leaf": [15, 40],
            "classifier__l2_regularization": [0.0, 1.0],
        }
    if model_type == "catboost":
        return {
            "classifier__iterations": [800, 1600],
            "classifier__learning_rate": [0.03, 0.08],
            "classifier__depth": [6, 10],
            "classifier__l2_leaf_reg": [2.0, 5.0],
        }
    if model_type == "xgboost":
        return {
            "classifier__n_estimators": [400, 600],
            "classifier__learning_rate": [0.03, 0.08],
            "classifier__max_depth": [6, 10],
            "classifier__subsample": [0.7, 0.9],
            "classifier__colsample_bytree": [0.7, 0.9],
            "classifier__reg_lambda": [0.5, 1.5],
            "classifier__min_child_weight": [1, 5],
        }
    return {}


def run_hyperparameter_search(
    *,
    estimator: Pipeline,
    param_distributions: Dict[str, Sequence[Any]],
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int,
    cv_splits: int,
    random_state: int,
    n_jobs: int | None,
    sample_weight: np.ndarray | None,
) -> Dict[str, Any]:
    """Run a randomized hyperparameter search and return the best configuration."""
    stratified_cv = StratifiedKFold(
        n_splits=cv_splits, shuffle=True, random_state=random_state
    )

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="f1_weighted",
        cv=stratified_cv,
        random_state=random_state,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=False,
    )

    fit_kwargs: Dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["classifier__sample_weight"] = sample_weight

    search.fit(X, y, **fit_kwargs)
    return {
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
    }


def run_cross_validation(
    pipeline_builder,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int,
    random_state: int,
) -> Dict[str, Any]:
    """Run stratified K-fold CV and gather evaluation artefacts."""
    stratified_cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    fold_scores: List[Dict[str, Any]] = []
    y_true_chunks: List[pd.Series] = []
    y_pred_chunks: List[pd.Series] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(
        stratified_cv.split(X, y), start=1
    ):
        pipeline = pipeline_builder()
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        pipeline.fit(
            X_train,
            y_train,
            classifier__sample_weight=sample_weight,
        )

        y_pred = pipeline.predict(X_valid)
        fold_f1_weighted = f1_score(y_valid, y_pred, average="weighted")

        fold_scores.append(
            {
                "fold": fold_idx,
                "f1_weighted": float(fold_f1_weighted),
            }
        )
        y_true_chunks.append(y_valid)
        y_pred_chunks.append(pd.Series(y_pred, index=y_valid.index))

    y_true_all = pd.concat(y_true_chunks).sort_index()
    y_pred_all = pd.concat(y_pred_chunks).reindex(y_true_all.index)

    overall_f1 = f1_score(y_true_all, y_pred_all, average="weighted")
    unique_labels = sorted(pd.unique(y_true_all))
    normalized_labels = [_normalize_label(label) for label in unique_labels]
    confusion = confusion_matrix(y_true_all, y_pred_all, labels=normalized_labels)
    classif_report = classification_report(
        y_true_all, y_pred_all, output_dict=True, zero_division=0
    )

    fold_values = [fold["f1_weighted"] for fold in fold_scores]
    cv_std = float(np.std(fold_values, ddof=1)) if len(fold_values) > 1 else 0.0
    summary = {
        "folds": fold_scores,
        "cv_f1_weighted_mean": float(np.mean(fold_values)),
        "cv_f1_weighted_std": cv_std,
        "overall_f1_weighted": float(overall_f1),
        "classification_report": classif_report,
        "confusion_matrix": confusion.tolist(),
        "class_labels": [_normalize_label(label) for label in normalized_labels],
    }
    return summary


def train_final_model(
    pipeline_builder,
    X: pd.DataFrame,
    y: pd.Series,
) -> Pipeline:
    """Fit the final model on the full dataset."""
    pipeline = pipeline_builder()
    sample_weight = compute_sample_weight(class_weight="balanced", y=y)
    pipeline.fit(X, y, classifier__sample_weight=sample_weight)
    return pipeline


def ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train booking status model with stratified cross-validation."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to the training data CSV (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default=DEFAULT_TARGET_COLUMN,
        help=f"Name of the target column (default: {DEFAULT_TARGET_COLUMN})",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of folds for Stratified K-Fold cross-validation (default: 5).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=3,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--model",
        choices=["histgb", "catboost", "xgboost"],
        default="xgboost",
        help="Which classifier to train (default: catboost).",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Enable randomized hyperparameter search before cross-validation.",
    )
    parser.add_argument(
        "--search-iterations",
        type=int,
        default=15,
        help="Number of parameter settings sampled during search (default: 15).",
    )
    parser.add_argument(
        "--search-n-splits",
        type=int,
        default=5,
        help="Number of CV folds used during hyperparameter search (default: --n-splits).",
    )
    parser.add_argument(
        "--search-n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for hyperparameter search (-1 uses all cores, default: 1).",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help=f"Where to dump evaluation metrics as JSON (default: {DEFAULT_METRICS_PATH}).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Where to persist the trained pipeline (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip persisting the trained pipeline and metrics to disk.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.data_path.exists():
        raise FileNotFoundError(f"Training data not found at {args.data_path}")

    df = pd.read_csv(args.data_path)
    if args.target_column not in df.columns:
        raise ValueError(
            f"Target column '{args.target_column}' missing from dataset columns."
        )

    X = df.drop(columns=[args.target_column])
    y = df[args.target_column]

    base_pipeline = build_pipeline(
        random_state=args.random_state,
        model_type=args.model,
    )
    features_step = base_pipeline.named_steps.get("features")
    removed_outliers = 0
    if (
        isinstance(features_step, DataFrameTransformer)
        and features_step.drop_adr_outliers
    ):
        upper_bound = getattr(features_step, "adr_upper_bound", None)
        if upper_bound is not None:
            X, y, removed_outliers = _drop_adr_outliers(
                X,
                y,
                upper_bound=upper_bound,
            )
            if removed_outliers:
                print(
                    f"Dropped {removed_outliers} rows with adr above {upper_bound} before training."
                )
    del base_pipeline

    tuned_params: Dict[str, Any] = {}
    search_metadata: Dict[str, Any] | None = None
    if args.search:
        param_distributions = get_param_distributions(args.model)
        if not param_distributions:
            print(
                f"No hyperparameter search space configured for model '{args.model}'. Skipping search."
            )
        else:
            search_cv_splits = args.search_n_splits or args.n_splits
            search_pipeline = build_pipeline(
                random_state=args.random_state,
                model_type=args.model,
            )
            sample_weight_full = compute_sample_weight(
                class_weight="balanced",
                y=y,
            )
            search_metadata = run_hyperparameter_search(
                estimator=search_pipeline,
                param_distributions=param_distributions,
                X=X,
                y=y,
                n_iter=args.search_iterations,
                cv_splits=search_cv_splits,
                random_state=args.random_state,
                n_jobs=args.search_n_jobs,
                sample_weight=sample_weight_full,
            )
            tuned_params = search_metadata["best_params"]
            print("Hyperparameter search results:")
            print(f"  Best mean CV f1_weighted: {search_metadata['best_score']:.4f}")
            for param_name, value in sorted(tuned_params.items()):
                print(f"  {param_name} = {value}")

    def pipeline_builder() -> Pipeline:
        pipeline = build_pipeline(
            random_state=args.random_state,
            model_type=args.model,
        )
        if tuned_params:
            pipeline.set_params(**tuned_params)
        return pipeline

    metrics = run_cross_validation(
        pipeline_builder,
        X,
        y,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )
    metrics["model_type"] = args.model
    metrics["n_splits"] = args.n_splits
    if search_metadata is not None:
        search_cv_splits = args.search_n_splits or args.n_splits
        metrics["hyperparameter_search"] = {
            "best_params": search_metadata["best_params"],
            "best_score": search_metadata["best_score"],
            "n_iter": args.search_iterations,
            "n_splits": search_cv_splits,
        }

    print(f"Model type: {args.model}")
    print("Cross-validation results:")
    for fold in metrics["folds"]:
        print(f"  Fold {fold['fold']}: F1_weighted = {fold['f1_weighted']:.4f}")
    print(
        f"  Mean F1_weighted: {metrics['cv_f1_weighted_mean']:.4f} "
        f"(±{metrics['cv_f1_weighted_std']:.4f})"
    )
    print(f"  Overall (OOF) F1_weighted: {metrics['overall_f1_weighted']:.4f}")

    pipeline = train_final_model(pipeline_builder, X, y)

    if not args.no_save:
        ensure_directory(args.model_path)
        joblib.dump(pipeline, args.model_path)

        ensure_directory(args.metrics_path)
        with args.metrics_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2, default=_json_default)

        print(f"Saved trained pipeline to {args.model_path}")
        print(f"Saved metrics to {args.metrics_path}")


if __name__ == "__main__":
    main()
