import argparse
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, ElasticNet
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

!pip install factor_analyzer

try:
    from factor_analyzer import FactorAnalyzer
except Exception as exc:  # pragma: no cover
    FactorAnalyzer = None
    FACTOR_ANALYZER_IMPORT_ERROR = exc
else:
    FACTOR_ANALYZER_IMPORT_ERROR = None

from sklearn.base import clone
from sklearn.model_selection import GroupKFold, KFold, ParameterSampler

RANDOM_STATE = 42
OBSERVATION_YEAR = 2024

TARGET_COL = "Business_Status"
ID_COL = "ID"
SURVIVAL_CLASS = 0  # 0 = Active/survived in your Phase 1 script.
FAILURE_CLASS = 1

TARGET_DEFINITION = {"0": "Active/survived", "1": "Failed"}

RAW_REVENUE_COLS = [
    "Total_Revenue_2020",
    "Total_Revenue_2021",
    "Total_Revenue_2022",
    "Total_Revenue_2023",
]

RAW_EXPENDITURE_COLS = [
    "Total_expenditure_2020",
    "Total_expenditure_2021",
    "Total_expenditure_2022",
    "Total_expenditure_2023",
]

RAW_STAFF_COLS = [
    "Staff_total_2020",
    "Staff_total_2021",
    "Staff_total_2022",
    "Staff_total_2023",
]

RAW_PHASE1_INPUT_COLS = RAW_STAFF_COLS + RAW_EXPENDITURE_COLS + RAW_REVENUE_COLS
REQUIRED_DATASET_COLS = [ID_COL, TARGET_COL] + RAW_PHASE1_INPUT_COLS

DROP_FROM_PHASE2 = [
    TARGET_COL,
    "Unnamed: 0",
    "Open_Year",
    "Active_upto_year",
    "Total_Revenue_2020",
    "Total_Revenue_2021",
    "Total_Revenue_2022",
    "Total_Revenue_2023",
    "Total_expenditure_2020",
    "Total_expenditure_2021",
    "Total_expenditure_2022",
    "Total_expenditure_2023",
    "Staff_total_2020",
    "Staff_total_2021",
    "Staff_total_2022",
    "Staff_total_2023"
]

OWNER_HOUSEHOLD_VARS = [
    "Owner_age",
    "Owner_family_size",
]

BUSINESS_SIZE_STAFF_VARS = [
    "Owners_2020", "Owners_2021", "Owners_2022", "Owners_2023",
    "Admin_Staff_2020", "Admin_Staff_2021", "Admin_Staff_2022", "Admin_Staff_2023",
    "Tech_Staff_2020", "Tech_Staff_2021", "Tech_Staff_2022", "Tech_Staff_2023",
]

COST_STRUCTURE_VARS = [
    "Employee_compensation_2020", "Employee_compensation_2021",
    "Employee_compensation_2022", "Employee_compensation_2023",
    "Cost_of_raw_materials_2020", "Cost_of_raw_materials_2021",
    "Cost_of_raw_materials_2022", "Cost_of_raw_materials_2023",
    "Utility_bills_2020", "Utility_bills_2021",
    "Utility_bills_2022", "Utility_bills_2023",
    "Rent_2020", "Rent_2021", "Rent_2022", "Rent_2023",
    "Transport_cost_2020", "Transport_cost_2021",
    "Transport_cost_2022", "Transport_cost_2023",
    "Maintenance_cost_2020", "Maintenance_cost_2021",
    "Maintenance_cost_2022", "Maintenance_cost_2023",
    "Other_expenses_2020", "Other_expenses_2021",
    "Other_expenses_2022", "Other_expenses_2023",
]

BUSINESS_PERFORMANCE_VARS = [
    "Gross_annual_revenue_2020", "Gross_annual_revenue_2021",
    "Gross_annual_revenue_2022", "Gross_annual_revenue_2023",
    "Minimum_2023_Monthly_profit",
    "Maximum_2023_Monthly_profit",
    "Other_income_total_2020", "Other_income_total_2021",
    "Other_income_total_2022", "Other_income_total_2023",
]

INVESTMENT_DEBTS_VARS = [
    "Cash_inflow_investing_2020", "Cash_inflow_investing_2021",
    "Cash_inflow_investing_2022", "Cash_inflow_investing_2023",
    "Cash_inflow_financial_activities_2020",
    "Cash_inflow_financial_activities_2021",
    "Cash_inflow_financial_activities_2022",
    "Cash_inflow_financial_activities_2023",
    "total_investment_2020", "total_investment_2021",
    "total_investment_2022", "total_investment_2023",
    "non-performing_debt_2020", "non-performing_debt_2021",
    "non-performing_debt_2022", "non-performing_debt_2023",
]

FA_SPECS = {
    "staff": {
        "variables": BUSINESS_SIZE_STAFF_VARS,
        "n_factors": 3,
        "score_columns": [
            "FA_Technical_Staff_Capacity",
            "FA_Admin_Staff_Capacity",
            "FA_Owner_Capacity",
        ],
    },
    "cost": {
        "variables": COST_STRUCTURE_VARS,
        "n_factors": 7,
        "score_columns": [
            "FA_Recent_Raw_Material_and_Utility_Cost",
            "FA_Employee_Compensation_Cost",
            "FA_Rental_Cost_Burden",
            "FA_Transport_Cost_Burden",
            "FA_Other_Operating_Expenses",
            "FA_Maintenance_Cost_Burden",
            "FA_Early_Raw_Material_and_Utility_Cost"


        ],
    },
    "performance": {
        "variables": BUSINESS_PERFORMANCE_VARS,
        "n_factors": 3,
        "score_columns": [
            "FA_Other_Income_total",
            "FA_Gross_Revenue",
            "FA_Min_Max_Profit_latest",
        ],
    },
    "investment": {
        "variables": INVESTMENT_DEBTS_VARS,
        "n_factors": 4,
        "score_columns": [
            "FA_Cash_inflow_financial_activities",
            "FA_Cash_inflow_investing",
            "FA_Non_performing_Debt",
            "FA_total_investment",
        ],
    },
}

def validate_binary_target(y: pd.Series, source_name: str = TARGET_COL) -> pd.Series:
    y_numeric = pd.to_numeric(y, errors="coerce")
    if y_numeric.isna().any():
        raise ValueError(f"{source_name} contains missing or non-numeric values.")
    y_int = y_numeric.astype(int)
    invalid_values = sorted(set(y_int.unique()) - {0, 1})
    if invalid_values:
        raise ValueError(f"{source_name} must contain only 0 and 1. Found: {invalid_values}")
    return y_int

def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    '''
    Read CSV file
    ↓
    Clean column names
    ↓
    Check required columns exist
    ↓
    Validate Business_Status
    ↓
    Remove exact duplicate rows
    ↓
    Confirm both classes exist
    ↓
    Confirm ID column has no missing values
    ↓
    Return cleaned dataset
    '''

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    missing_cols = [col for col in REQUIRED_DATASET_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required column(s): {missing_cols}")

    df[TARGET_COL] = validate_binary_target(df[TARGET_COL])
    df = df.drop_duplicates().reset_index(drop=True)

    if df[TARGET_COL].nunique() < 2:
        raise ValueError("The dataset must contain both classes: 0 = Active and 1 = Failed.")

    if df[ID_COL].isna().any():
        raise ValueError(f"{ID_COL} contains missing values. Grouped splitting requires valid IDs.")

    return df

def make_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create one split used by both Phase 1 and Phase 2."""
    y = validate_binary_target(df[TARGET_COL])
    groups = df[ID_COL].astype(str)

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(outer.split(df, y, groups))

    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    y_train_val = validate_binary_target(train_val_df[TARGET_COL])
    groups_train_val = train_val_df[ID_COL].astype(str)

    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE + 1)
    train_idx, val_idx = next(inner.split(train_val_df, y_train_val, groups_train_val))

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    for split_name, split_df in {"train": train_df, "validation": val_df, "test": test_df}.items():
        if split_df[TARGET_COL].nunique() < 2:
            raise ValueError(f"The {split_name} split contains only one target class.")

    return train_df, val_df, test_df

def safe_metric(metric_func, default: float = np.nan, **kwargs) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            return float(metric_func(**kwargs))
    except Exception:
        return float(default)

def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def safe_divide_frame(numerator: pd.DataFrame, denominator: pd.DataFrame) -> pd.DataFrame:
    denominator_values = denominator.to_numpy(dtype=float)
    denominator_values = np.where(denominator_values == 0, np.nan, denominator_values)
    divided = numerator.to_numpy(dtype=float) / denominator_values
    return pd.DataFrame(divided, index=numerator.index, columns=numerator.columns)

def last_observed_df(frame: pd.DataFrame) -> pd.Series:
    values = frame.to_numpy(dtype=float)
    out = np.full(values.shape[0], np.nan, dtype=float)
    for i, row in enumerate(values):
        valid = row[~np.isnan(row)]
        if valid.size:
            out[i] = valid[-1]
    return pd.Series(out, index=frame.index)

def trend_slope_df(frame: pd.DataFrame) -> pd.Series:
    x_full = np.arange(frame.shape[1], dtype=float).reshape(-1, 1)
    values = frame.to_numpy(dtype=float)
    out = np.full(values.shape[0], np.nan, dtype=float)
    for i, row in enumerate(values):
        mask = ~np.isnan(row)
        if mask.sum() < 2:
            continue
        model = LinearRegression()
        model.fit(x_full[mask], row[mask])
        out[i] = float(model.coef_[0])
    return pd.Series(out, index=frame.index)

def build_phase1_feature_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    revenue = raw_df[RAW_REVENUE_COLS].apply(pd.to_numeric, errors="coerce")
    expenditure = raw_df[RAW_EXPENDITURE_COLS].apply(pd.to_numeric, errors="coerce")
    staff = raw_df[RAW_STAFF_COLS].apply(pd.to_numeric, errors="coerce")

    profit = pd.DataFrame(
        revenue.to_numpy(dtype=float) - expenditure.to_numpy(dtype=float),
        columns=["Profit_2020", "Profit_2021", "Profit_2022", "Profit_2023"],
        index=raw_df.index,
    )
    margin = safe_divide_frame(profit, revenue)
    margin.columns = ["Margin_2020", "Margin_2021", "Margin_2022", "Margin_2023"]

    features = pd.DataFrame(index=raw_df.index)
    features["Revenue_latest"] = last_observed_df(revenue)
    features["Revenue_mean"] = revenue.mean(axis=1, skipna=True)
    features["Revenue_std"] = revenue.std(axis=1, skipna=True)
    features["Expenditure_latest"] = last_observed_df(expenditure)
    features["Expenditure_mean"] = expenditure.mean(axis=1, skipna=True)
    features["Expenditure_std"] = expenditure.std(axis=1, skipna=True)
    features["Staff_latest"] = last_observed_df(staff)
    features["Staff_mean"] = staff.mean(axis=1, skipna=True)
    features["Staff_std"] = staff.std(axis=1, skipna=True)
    features["Profit_latest"] = last_observed_df(profit)
    features["Profit_mean"] = profit.mean(axis=1, skipna=True)
    features["Profit_std"] = profit.std(axis=1, skipna=True)
    features["Profit_margin_latest"] = last_observed_df(margin)
    features["Profit_margin_mean"] = margin.mean(axis=1, skipna=True)
    features["Revenue_trend"] = trend_slope_df(revenue)
    features["Expenditure_trend"] = trend_slope_df(expenditure)
    features["Staff_trend"] = trend_slope_df(staff)
    features["Loss_years"] = (profit < 0).sum(axis=1)
    return features

def build_phase1_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X_phase1 = build_phase1_feature_frame(df[RAW_PHASE1_INPUT_COLS].copy())
    y = validate_binary_target(df[TARGET_COL])
    return X_phase1, y

def make_phase1_model_search_spaces(scale_pos_weight: float) -> Dict[str, Tuple[Pipeline, Dict[str, List[Any]]]]:
    searches: Dict[str, Tuple[Pipeline, Dict[str, List[Any]]]] = {}

    searches["Logistic Regression"] = (
        Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        {"model__C": np.logspace(-3, 2, 12).tolist()},
    )

    searches["Random Forest"] = (
        Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 0.8],
        },
    )

    if xgb is not None:
        searches["XGBoost"] = (
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        xgb.XGBClassifier(
                            random_state=RANDOM_STATE,
                            eval_metric="logloss",
                            tree_method="hist",
                            scale_pos_weight=scale_pos_weight,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            {
                "model__n_estimators": [200, 400, 600],
                "model__max_depth": [3, 4, 5, 6],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
                "model__min_child_weight": [1, 3, 5],
            },
        )

    if lgb is not None:
        searches["LightGBM"] = (
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        lgb.LGBMClassifier(
                            random_state=RANDOM_STATE,
                            class_weight="balanced",
                            verbose=-1,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            {
                "model__n_estimators": [200, 400, 600],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__num_leaves": [15, 31, 63],
                "model__max_depth": [-1, 5, 10, 20],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
                "model__min_child_samples": [10, 20, 40],
            },
        )

    return searches

def tune_phase1_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    n_iter: int,
) -> Dict[str, Dict[str, Any]]:

    '''
    Count active and failed businesses
    ↓
    Calculate imbalance weight for models like XGBoost
    ↓
    Create Phase 1 model search spaces
    ↓
    Use 5-fold stratified grouped cross-validation
    ↓
    Tune each Phase 1 model using RandomizedSearchCV
    ↓
    Store the best model, best parameters, and best CV score for each model
    ↓
    Return all tuned model results
    '''

    positive = int((y_train == FAILURE_CLASS).sum())
    negative = int((y_train == SURVIVAL_CLASS).sum())
    scale_pos_weight = negative / positive if positive else 1.0

    searches = make_phase1_model_search_spaces(scale_pos_weight)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    groups = groups_train.astype(str).reset_index(drop=True)

    tuned: Dict[str, Dict[str, Any]] = {}
    for model_name, (pipeline, param_grid) in searches.items():
        print(f"\nTuning Phase 1 model: {model_name}")
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring="f1_macro",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=1,
            refit=True,
        )
        search.fit(X_train, y_train, groups=groups)
        tuned[model_name] = {
            "search": search,
            "best_estimator": search.best_estimator_,
            "best_params": search.best_params_,
            "best_cv_score": float(search.best_score_),
        }
    return tuned

def evaluate_phase1_classifier(model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    y_true = np.asarray(y).astype(int)
    y_pred = model.predict(X)
    p_failed = get_probability_for_class(model, X, FAILURE_CLASS)
    p_survival = get_probability_for_class(model, X, SURVIVAL_CLASS)

    return {
        "accuracy": safe_metric(accuracy_score, y_true=y_true, y_pred=y_pred),
        "balanced_accuracy": safe_metric(balanced_accuracy_score, y_true=y_true, y_pred=y_pred),
        "precision_macro": safe_metric(precision_score, y_true=y_true, y_pred=y_pred, average="macro", zero_division=0),
        "recall_macro": safe_metric(recall_score, y_true=y_true, y_pred=y_pred, average="macro", zero_division=0),
        "f1_macro": safe_metric(f1_score, y_true=y_true, y_pred=y_pred, average="macro", zero_division=0),
        "roc_auc_failed": safe_metric(roc_auc_score, y_true=y_true, y_score=p_failed),
        "pr_auc_failed": safe_metric(average_precision_score, y_true=y_true, y_score=p_failed),
        "brier_failed": safe_metric(brier_score_loss, y_true=y_true, y_proba=p_failed, pos_label=FAILURE_CLASS),
        "brier_survival": safe_metric(brier_score_loss, y_true=(y_true == SURVIVAL_CLASS).astype(int), y_proba=p_survival),
    }

def select_best_phase1_model(
    tuned_models: Dict[str, Dict[str, Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, Any]:
    rows = []
    for name, info in tuned_models.items():
        fitted = clone(info["best_estimator"])
        fitted.fit(X_train, y_train)
        metrics = evaluate_phase1_classifier(fitted, X_val, y_val)
        rows.append({"model_name": name, "cv_f1_macro": info["best_cv_score"], **metrics})
        info["fitted_on_train"] = fitted
        info["validation_metrics"] = metrics

    comparison = pd.DataFrame(rows).sort_values(
        by=["f1_macro", "balanced_accuracy", "brier_survival"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    best_name = str(comparison.loc[0, "model_name"])
    return {
        "best_name": best_name,
        "best_info": tuned_models[best_name],
        "validation_table": comparison,
    }

def get_probability_for_class(model: Any, X: pd.DataFrame, class_label: int) -> np.ndarray:
    '''
    Ask the classifier for predicted probabilities
    ↓
    Check the model’s class order
    ↓
    Find the column for the class we want
    ↓
    Return that probability column
    '''
    probs = model.predict_proba(X)
    classes = [int(cls) for cls in model.classes_]
    if class_label not in classes:
        raise ValueError(f"Model classes are {classes}; class {class_label} was not found.")
    class_index = classes.index(class_label)
    return probs[:, class_index]

def get_oof_probabilities(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    class_label: int,
    n_splits: int = 5,
) -> np.ndarray:
    """OOF probabilities for Phase 2 training target.

    Each training row receives a Phase 1 probability from a model that did not
    train on that row. This avoids teaching Phase 2 in-sample Phase 1 noise.
    """

    '''
    Training data
    ↓
    Split into cross-validation folds
    ↓
    For each fold:
        train Phase 1 model on other folds
        predict survival probability for held-out fold
    ↓
    Every training row gets a Phase 1 probability from a model that did not train on it
    ↓
    Use those probabilities as the Phase 2 target
    '''
    y = validate_binary_target(y)
    min_class_count = int(y.value_counts().min())
    n_unique_groups = int(groups.astype(str).nunique())
    n_splits = min(n_splits, min_class_count, n_unique_groups)
    if n_splits < 2:
        raise ValueError("Not enough samples/groups/classes for out-of-fold probability generation.")

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE + 100)
    oof_probs = np.full(len(X), np.nan, dtype=float)

    X_reset = X.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)
    groups_reset = groups.astype(str).reset_index(drop=True)

    for fold, (fit_idx, pred_idx) in enumerate(cv.split(X_reset, y_reset, groups_reset), start=1):
        fold_model = clone(estimator)
        fold_model.fit(X_reset.iloc[fit_idx], y_reset.iloc[fit_idx])
        oof_probs[pred_idx] = get_probability_for_class(fold_model, X_reset.iloc[pred_idx], class_label)
        print(f"Generated Phase 1 OOF probabilities for fold {fold}/{n_splits}")

    if np.isnan(oof_probs).any():
        raise RuntimeError("Some out-of-fold probabilities were not generated.")

    return oof_probs

# -----------------------------------------------------------------------------
# Phase 2 factor-score feature builder
# -----------------------------------------------------------------------------

'''
Take the training dataset
↓
Remove columns not allowed in Phase 2
↓
Check required factor-analysis columns exist
↓
Remove ID from modelling features
↓
Separate categorical and continuous numeric variables
↓
Clean invalid numeric values
↓
Fit categorical imputer
↓
Fit numeric imputer and fill missing numeric values
↓
Detect highly skewed columns using training data only
↓
Log-transform those skewed columns
↓
Fit scaler and standardize continuous variables
↓
For each factor group:
    select the correct variables
    fit factor analysis
    store the fitted factor model
    store factor-score names
↓
Return the fitted feature builder
'''

@dataclass
class Phase2FactorFeatureBuilder:
    categorical_columns: List[str] = field(default_factory=list)
    numeric_columns: List[str] = field(default_factory=list)
    skewed_columns: List[str] = field(default_factory=list)
    imputer_cat: Optional[SimpleImputer] = None
    imputer_num: Optional[SimpleImputer] = None
    scaler: Optional[StandardScaler] = None
    factor_models: Dict[str, Any] = field(default_factory=dict)
    factor_score_columns: List[str] = field(default_factory=list)
    direct_continuous_columns: List[str] = field(default_factory=lambda: OWNER_HOUSEHOLD_VARS.copy())

    def _make_phase2_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = [col for col in DROP_FROM_PHASE2 if col in df.columns]
        return df.drop(columns=drop_cols)

    def _validate_required_factor_columns(self, X_raw: pd.DataFrame) -> None:
        required = []
        for spec in FA_SPECS.values():
            required.extend(spec["variables"])
        required.extend(self.direct_continuous_columns)
        missing = sorted(set(required) - set(X_raw.columns))
        if missing:
            raise ValueError(f"Missing Phase 2 factor/direct column(s): {missing}")

    @staticmethod
    def _clean_numeric_values(X_num: pd.DataFrame) -> pd.DataFrame:
        X_num = X_num.copy()
        if "Owner_age" in X_num.columns:
            X_num.loc[X_num["Owner_age"] == 0, "Owner_age"] = np.nan
            X_num["Owner_age"] = X_num["Owner_age"].astype(float)
        if "Owner_family_size" in X_num.columns:
            X_num.loc[X_num["Owner_family_size"] == 0, "Owner_family_size"] = np.nan
            X_num.loc[X_num["Owner_family_size"] > 15, "Owner_family_size"] = np.nan
            X_num["Owner_family_size"] = X_num["Owner_family_size"].astype(float)
        return X_num

    def fit(self, df: pd.DataFrame) -> "Phase2FactorFeatureBuilder":
        if FactorAnalyzer is None:
            raise ImportError(
                "factor_analyzer is not installed. Run `pip install factor_analyzer` first."
            ) from FACTOR_ANALYZER_IMPORT_ERROR

        X_raw = self._make_phase2_raw(df)
        self._validate_required_factor_columns(X_raw)

        # Keep ID separately; it is not a modeling feature.
        feature_cols = [col for col in X_raw.columns if col != ID_COL]
        X_features = X_raw[feature_cols].copy()

        numeric_candidate_cols = X_features.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = [
            col for col in numeric_candidate_cols if X_features[col].nunique(dropna=True) <= 2
        ]
        self.numeric_columns = [col for col in numeric_candidate_cols if col not in self.categorical_columns]

        X_cat = X_features[self.categorical_columns].copy()
        X_num = X_features[self.numeric_columns].apply(pd.to_numeric, errors="coerce")
        X_num = self._clean_numeric_values(X_num)

        self.imputer_cat = SimpleImputer(strategy="most_frequent")
        if len(self.categorical_columns) > 0:
            self.imputer_cat.fit(X_cat)

        self.imputer_num = SimpleImputer(strategy="mean")
        X_num_imputed = pd.DataFrame(
            self.imputer_num.fit_transform(X_num),
            columns=self.numeric_columns,
            index=df.index,
        )

        # Select skewed columns using training data only, then store the names.

        self.skewed_columns = []
        for col in X_num_imputed.columns:
            col_min = X_num_imputed[col].min(skipna=True)
            col_skew = X_num_imputed[col].skew(skipna=True)
            if col_min >= 0 and abs(col_skew) >= 2:
                self.skewed_columns.append(col)
                X_num_imputed[col] = np.log1p(X_num_imputed[col])

        self.scaler = StandardScaler()
        X_num_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_num_imputed),
            columns=self.numeric_columns,
            index=df.index,
        )

        self.factor_models = {}
        self.factor_score_columns = []
        for group_name, spec in FA_SPECS.items():
            variables = spec["variables"]
            n_factors = spec["n_factors"]
            score_columns = spec["score_columns"]

            missing = sorted(set(variables) - set(X_num_scaled.columns))
            if missing:
                raise ValueError(
                    f"Factor group '{group_name}' contains columns that were not treated as continuous numeric: {missing}"
                )

            subset = X_num_scaled[variables]
            fa = FactorAnalyzer(n_factors=n_factors, rotation="promax")
            fa.fit(subset)

            self.factor_models[group_name] = fa
            self.factor_score_columns.extend(score_columns)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.imputer_num is None or self.scaler is None:
            raise RuntimeError("Phase2FactorFeatureBuilder must be fitted before transform().")

        X_raw = self._make_phase2_raw(df)
        self._validate_required_factor_columns(X_raw)
        feature_cols = [col for col in X_raw.columns if col != ID_COL]
        X_features = X_raw[feature_cols].copy()

        # Enforce training columns and order.
        X_cat = X_features.reindex(columns=self.categorical_columns).copy()
        X_num = X_features.reindex(columns=self.numeric_columns).apply(pd.to_numeric, errors="coerce")
        X_num = self._clean_numeric_values(X_num)

        if len(self.categorical_columns) > 0:
            X_cat_imputed = pd.DataFrame(
                self.imputer_cat.transform(X_cat),
                columns=self.categorical_columns,
                index=df.index,
            )
        else:
            X_cat_imputed = pd.DataFrame(index=df.index)

        X_num_imputed = pd.DataFrame(
            self.imputer_num.transform(X_num),
            columns=self.numeric_columns,
            index=df.index,
        )

        for col in self.skewed_columns:
            if col in X_num_imputed.columns:
                X_num_imputed[col] = np.log1p(X_num_imputed[col].clip(lower=0))

        X_num_scaled = pd.DataFrame(
            self.scaler.transform(X_num_imputed),
            columns=self.numeric_columns,
            index=df.index,
        )

        factor_frames = []
        for group_name, spec in FA_SPECS.items():
            variables = spec["variables"]
            score_columns = spec["score_columns"]
            fa = self.factor_models[group_name]
            scores = fa.transform(X_num_scaled[variables])
            factor_frames.append(pd.DataFrame(scores, columns=score_columns, index=df.index))

        direct_cols_available = [c for c in self.direct_continuous_columns if c in X_num_scaled.columns]
        direct_frame = X_num_scaled[direct_cols_available].copy()
        direct_frame.columns = [f"scaled_{c}" for c in direct_frame.columns]

        X_phase2 = pd.concat(factor_frames + [direct_frame, X_cat_imputed], axis=1)
        return X_phase2

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

def make_phase2_regressor_search_spaces() -> Dict[str, Tuple[Pipeline, Dict[str, List[Any]]]]:
    spaces = {
        "Ridge": (
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", Ridge(random_state=RANDOM_STATE)),
                ]
            ),
            {
                "model__alpha": np.logspace(-3, 3, 20).tolist()
            },
        ),

        "Random Forest Regressor": (
            Pipeline(
                steps=[
                    ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
                ]
            ),
            {
                "model__n_estimators": [200, 400, 600],
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", 0.8],
            },
        ),

        "Hist Gradient Boosting Regressor": (
            Pipeline(
                steps=[
                    ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
                ]
            ),
            {
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__max_iter": [100, 200, 300],
                "model__max_leaf_nodes": [15, 31, 63],
                "model__l2_regularization": [0.0, 0.01, 0.1, 1.0],
            },
        ),

        "XGBoost Regressor": (
            Pipeline(
                steps=[
                    (
                        "model",
                        XGBRegressor(
                            objective="reg:squarederror",
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                            tree_method="hist",
                        ),
                    ),
                ]
            ),
            {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [2, 3, 4, 5],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__subsample": [0.7, 0.9, 1.0],
                "model__colsample_bytree": [0.7, 0.9, 1.0],
                "model__reg_lambda": [0.1, 1.0, 5.0, 10.0],
            },
        ),

        "LightGBM Regressor": (
            Pipeline(
                steps=[
                    (
                        "model",
                        LGBMRegressor(
                            objective="regression",
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                            verbose=-1,
                            importance_type="gain"
                        ),
                    ),
                ]
            ),
            {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__num_leaves": [15, 31, 63],
                "model__max_depth": [-1, 3, 5, 10],
                "model__min_child_samples": [10, 20, 30],
                "model__reg_lambda": [0.0, 0.1, 1.0, 5.0],
            },
        ),
    }

    return spaces

def tune_phase2_regressors(
    train_df: pd.DataFrame,
    y_train_probability: np.ndarray,
    groups_train: pd.Series,
    val_df: pd.DataFrame,
    y_val_probability: np.ndarray,
    n_iter: int,
) -> Dict[str, Any]:

    y_train_probability = np.asarray(y_train_probability, dtype=float)
    y_val_probability = np.asarray(y_val_probability, dtype=float)

    n_unique_groups = groups_train.astype(str).nunique()

    if n_unique_groups >= 5:
        cv = GroupKFold(n_splits=5)
        cv_groups = groups_train.astype(str).reset_index(drop=True)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_groups = None

    search_spaces = make_phase2_regressor_search_spaces()

    rows = []
    best_objects: Dict[str, Dict[str, Any]] = {}

    for model_name, (pipeline, param_grid) in search_spaces.items():
        print(f"\nTuning Phase 2 model without preprocessing leakage: {model_name}")

        param_candidates = list(
            ParameterSampler(
                param_grid,
                n_iter=n_iter,
                random_state=RANDOM_STATE,
            )
        )

        model_rows = []

        for params in param_candidates:
            fold_rmses = []

            split_iterator = cv.split(
                train_df,
                y_train_probability,
                groups=cv_groups,
            )

            for fold_train_idx, fold_valid_idx in split_iterator:
                fold_train_df = train_df.iloc[fold_train_idx].copy()
                fold_valid_df = train_df.iloc[fold_valid_idx].copy()

                y_fold_train = y_train_probability[fold_train_idx]
                y_fold_valid = y_train_probability[fold_valid_idx]

                # Important: fit Phase 2 preprocessing only on this fold's training part
                fold_builder = Phase2FactorFeatureBuilder()
                X_fold_train = fold_builder.fit_transform(fold_train_df)
                X_fold_valid = fold_builder.transform(fold_valid_df)

                fold_model = clone(pipeline)
                fold_model.set_params(**params)
                fold_model.fit(X_fold_train, y_fold_train)

                fold_pred = np.clip(fold_model.predict(X_fold_valid), 0, 1)
                fold_rmses.append(rmse(y_fold_valid, fold_pred))

            mean_cv_rmse = float(np.mean(fold_rmses))

            model_rows.append(
                {
                    "params": params,
                    "cv_rmse": mean_cv_rmse,
                }
            )

        best_candidate = min(model_rows, key=lambda row: row["cv_rmse"])
        best_params = best_candidate["params"]
        best_cv_rmse = best_candidate["cv_rmse"]

        # Final refit: now it is okay to fit the builder on the full training set
        final_builder = Phase2FactorFeatureBuilder()
        X_train_final = final_builder.fit_transform(train_df)
        X_val_final = final_builder.transform(val_df)

        final_model = clone(pipeline)
        final_model.set_params(**best_params)
        final_model.fit(X_train_final, y_train_probability)

        val_pred = np.clip(final_model.predict(X_val_final), 0, 1)

        metrics = {
            "validation_rmse": rmse(y_val_probability, val_pred),
            "validation_mae": float(mean_absolute_error(y_val_probability, val_pred)),
            "validation_r2": safe_metric(
                r2_score,
                y_true=y_val_probability,
                y_pred=val_pred,
            ),
        }

        rows.append(
            {
                "model_name": model_name,
                "best_cv_rmse": best_cv_rmse,
                **metrics,
                "best_params": best_params,
            }
        )

        best_objects[model_name] = {
            "best_estimator": final_model,
            "phase2_feature_builder": final_builder,
            "best_params": best_params,
            "best_cv_rmse": best_cv_rmse,
            "validation_metrics": metrics,
            "X_train_final": X_train_final,
            "X_val_final": X_val_final,
        }

    comparison = pd.DataFrame(rows).sort_values(
        by=["validation_rmse", "validation_mae"],
        ascending=[True, True],
    ).reset_index(drop=True)

    best_name = str(comparison.loc[0, "model_name"])

    return {
        "best_name": best_name,
        "best_info": best_objects[best_name],
        "validation_table": comparison,
    }

def make_prediction_output(
    split_name: str,
    df_split: pd.DataFrame,
    phase1_survival_probability_target: np.ndarray,
    phase2_survival_probability: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": split_name,
            ID_COL: df_split[ID_COL].values,
            TARGET_COL: df_split[TARGET_COL].values,
            "phase1_survival_probability_target": np.asarray(phase1_survival_probability_target, dtype=float),
            "phase2_survival_probability": np.clip(np.asarray(phase2_survival_probability, dtype=float), 0, 1),
        }
    )

csv_path= "/content/drive/MyDrive/MSME_data/final_dataset.csv"
output_dir = "/content/drive/MyDrive/MSME_data/msme_outputs"
phase1_n_iter = 20
phase2_n_iter = 20


# PHASE 1
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

df = load_dataset(csv_path)

train_df, val_df, test_df = make_splits(df)

X1_train, y_train = build_phase1_xy(train_df)
X1_val, y_val = build_phase1_xy(val_df)
X1_test, y_test = build_phase1_xy(test_df)

tuned_phase1 = tune_phase1_models(X1_train, y_train, train_df[ID_COL], n_iter=phase1_n_iter)
phase1_selection = select_best_phase1_model(tuned_phase1, X1_train, y_train, X1_val, y_val)
phase1_best_name = phase1_selection["best_name"]
phase1_best_estimator = phase1_selection["best_info"]["best_estimator"]
phase_1_val_metrics = evaluate_phase1_classifier(phase1_best_estimator, X1_val, y_val)
phase1_test_metrics = evaluate_phase1_classifier(phase1_best_estimator, X1_test, y_test)

# OOF survival probability for Phase 2 training target.
phase1_surv_prob_train = get_oof_probabilities(
    estimator=phase1_best_estimator,
    X=X1_train,
    y=y_train,
    groups=train_df[ID_COL],
    class_label=SURVIVAL_CLASS,
    n_splits=5,
)

# One fitted Phase 1 model trained on the training set is used to create
# validation/test Phase 2 targets.
phase1_target_model = clone(phase1_best_estimator)
phase1_target_model.fit(X1_train, y_train)
phase1_surv_prob_val = get_probability_for_class(phase1_target_model, X1_val, SURVIVAL_CLASS)
phase1_surv_prob_test = get_probability_for_class(phase1_target_model, X1_test, SURVIVAL_CLASS)

phase1_validation_table = phase1_selection["validation_table"]
phase1_validation_table

phase1_test_metrics

#PHASE 2
phase2_selection = tune_phase2_regressors(
    train_df=train_df,
    y_train_probability=phase1_surv_prob_train,
    groups_train=train_df[ID_COL],
    val_df=val_df,
    y_val_probability=phase1_surv_prob_val,
    n_iter=phase2_n_iter,
)

phase2_best_name = phase2_selection["best_name"]
phase2_model = phase2_selection["best_info"]["best_estimator"]
phase2_feature_builder = phase2_selection["best_info"]["phase2_feature_builder"]

X2_train = phase2_selection["best_info"]["X_train_final"]
X2_val = phase2_selection["best_info"]["X_val_final"]
X2_test = phase2_feature_builder.transform(test_df)

phase2_pred_train = np.clip(phase2_model.predict(X2_train), 0, 1)
phase2_pred_val = np.clip(phase2_model.predict(X2_val), 0, 1)
phase2_pred_test = np.clip(phase2_model.predict(X2_test), 0, 1)

phase2_metrics = {
        "train_rmse_against_phase1_oof_target": rmse(phase1_surv_prob_train, phase2_pred_train),
        "train_mae_against_phase1_oof_target": float(mean_absolute_error(phase1_surv_prob_train, phase2_pred_train)),
        "val_rmse_against_phase1_target": rmse(phase1_surv_prob_val, phase2_pred_val),
        "val_mae_against_phase1_target": float(mean_absolute_error(phase1_surv_prob_val, phase2_pred_val)),
        "val_r2_against_phase1_target": safe_metric(r2_score, y_true=phase1_surv_prob_val, y_pred=phase2_pred_val),
        "test_rmse_against_phase1_target": rmse(phase1_surv_prob_test, phase2_pred_test),
        "test_mae_against_phase1_target": float(mean_absolute_error(phase1_surv_prob_test, phase2_pred_test)),
        "test_r2_against_phase1_target": safe_metric(r2_score, y_true=phase1_surv_prob_test, y_pred=phase2_pred_test),
    }

# Optional diagnostic: compare Phase 2 probability against the real observed status.
# This is not the training target for Phase 2, but it is useful to report.
y_test_survived = (y_test.values == SURVIVAL_CLASS).astype(int)
phase2_metrics.update(
    {
        "diagnostic_test_brier_vs_actual_survival": safe_metric(
            brier_score_loss, y_true=y_test_survived, y_proba=phase2_pred_test
        ),
        "diagnostic_test_roc_auc_vs_actual_survival": safe_metric(
            roc_auc_score, y_true=y_test_survived, y_score=phase2_pred_test
        ),
    }
)

train_output = make_prediction_output("train", train_df, phase1_surv_prob_train, phase2_pred_train)
val_output = make_prediction_output("validation", val_df, phase1_surv_prob_val, phase2_pred_val)
test_output = make_prediction_output("test", test_df, phase1_surv_prob_test, phase2_pred_test)
all_predictions = pd.concat([train_output, val_output, test_output], axis=0).reset_index(drop=True)

phase1_selection["validation_table"].to_csv(output_dir / "phase1_validation_ranking.csv", index=False)
phase2_selection["validation_table"].to_csv(output_dir / "phase2_validation_ranking.csv", index=False)
all_predictions.to_csv(output_dir / "phase2_survival_probability_predictions.csv", index=False)

bundle = {
    "phase1_target_model": phase1_target_model,
    "phase1_best_model_name": phase1_best_name,
    "phase1_feature_columns": X1_train.columns.tolist(),
    "phase2_feature_builder": phase2_feature_builder,
    "phase2_model": phase2_model,
    "phase2_best_model_name": phase2_best_name,
    "phase2_feature_columns": X2_train.columns.tolist(),
    "target_definition": TARGET_DEFINITION,
    "survival_class": SURVIVAL_CLASS,
    "note": (
        "Phase 2 is trained as a regression model. Its target is the Phase 1 predicted "
        "survival probability, not Business_Status directly."
    ),
}
joblib.dump(bundle, output_dir / "merged_phase1_phase2_bundle.joblib")

summary = {
    "data": {
        "rows_after_exact_dedup": int(len(df)),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "active_survived_count_total": int((df[TARGET_COL] == SURVIVAL_CLASS).sum()),
        "failed_count_total": int((df[TARGET_COL] == FAILURE_CLASS).sum()),
    },
    "target_definition": TARGET_DEFINITION,
    "phase1": {
        "best_model_name": phase1_best_name,
        "best_params": phase1_selection["best_info"]["best_params"],
        "validation_metrics_for_phase1_target_model": phase_1_val_metrics,
        "test_metrics_for_phase1_target_model": phase1_test_metrics,
        "survival_probability_column": "phase1_survival_probability_target",
        "survival_probability_mean_train": float(np.mean(phase1_surv_prob_train)),
        "survival_probability_mean_validation": float(np.mean(phase1_surv_prob_val)),
        "survival_probability_mean_test": float(np.mean(phase1_surv_prob_test)),
    },
    "phase2": {
        "best_model_name": phase2_best_name,
        "best_params": phase2_selection["best_info"]["best_params"],
        "metrics": phase2_metrics,
        "n_phase2_features": int(X2_train.shape[1]),
        "factor_score_columns": phase2_feature_builder.factor_score_columns,
        "direct_continuous_columns": [f"scaled_{c}" for c in phase2_feature_builder.direct_continuous_columns],
        "categorical_columns": phase2_feature_builder.categorical_columns,
        "skewed_columns_selected_from_training_only": phase2_feature_builder.skewed_columns,
    },
    "outputs": {
        "predictions_csv": str(output_dir / "phase2_survival_probability_predictions.csv"),
        "bundle_joblib": str(output_dir / "merged_phase1_phase2_bundle.joblib"),
        "phase1_validation_ranking_csv": str(output_dir / "phase1_validation_ranking.csv"),
        "phase2_validation_ranking_csv": str(output_dir / "phase2_validation_ranking.csv"),
    },
}

with open(output_dir / "merged_training_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nMerged Phase 1 -> Phase 2 training complete.")
print(f"Phase 1 best model: {phase1_best_name}")
print(f"Phase 2 best model: {phase2_best_name}")
print("\nPhase 2 validation ranking:")
print(phase2_selection["validation_table"].to_string(index=False))
print("\nPhase 2 metrics:")
for key, value in phase2_metrics.items():
    print(f"  {key}: {value:.4f}")
print(f"\nSaved predictions to: {output_dir / 'phase2_survival_probability_predictions.csv'}")
print(f"Saved model bundle to: {output_dir / 'merged_phase1_phase2_bundle.joblib'}")
print(f"Saved summary to: {output_dir / 'merged_training_summary.json'}")

lgbm_model = phase2_model.named_steps["model"]

importance_df = pd.DataFrame({
    "feature": X2_train.columns,
    "importance": lgbm_model.feature_importances_
}).sort_values("importance", ascending=False).reset_index(drop=True)

display(importance_df.head(30))

