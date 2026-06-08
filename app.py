from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys
import types
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

try:
    from factor_analyzer import FactorAnalyzer
except Exception as exc:  # pragma: no cover
    FactorAnalyzer = None
    FACTOR_ANALYZER_IMPORT_ERROR = exc
else:
    FACTOR_ANALYZER_IMPORT_ERROR = None


# -----------------------------------------------------------------------------
# Constants copied from the merged Phase 1 -> Phase 2 training pipeline.
# These are needed so that the saved Phase2FactorFeatureBuilder can transform
# new dashboard inputs in exactly the same way as during training.
# -----------------------------------------------------------------------------

RANDOM_STATE = 42
OBSERVATION_YEAR = 2024

TARGET_COL = "Business_Status"
ID_COL = "ID"
SURVIVAL_CLASS = 0
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

DROP_FROM_PHASE2 = [
    TARGET_COL,
    "Unnamed: 0",
    "Open_Year",
    "Active_upto_year",
    *RAW_STAFF_COLS,
    *RAW_EXPENDITURE_COLS,
    *RAW_REVENUE_COLS,
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
            "FA_Maintenance_Cost_Burden",
            "FA_Early_Raw_Material_and_Utility_Cost",
            "FA_Transport_Cost_Burden",
            "FA_Other_Operating_Expenses",
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

INPUT_GROUPS = {
    "Owner / household": OWNER_HOUSEHOLD_VARS,
    "Business size and staff": BUSINESS_SIZE_STAFF_VARS,
    "Cost structure": COST_STRUCTURE_VARS,
    "Business performance": BUSINESS_PERFORMANCE_VARS,
    "Investment and debt": INVESTMENT_DEBTS_VARS,
}


# -----------------------------------------------------------------------------
# Compatibility copy of the Phase 2 feature builder.
# This allows joblib to reload bundles saved from a notebook or from the training
# script, then transform new dashboard rows using the fitted builder object.
# -----------------------------------------------------------------------------

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
        required: List[str] = []
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
        raise RuntimeError(
            "This dashboard only uses the fitted Phase2FactorFeatureBuilder saved in the model bundle. "
            "Fit the builder in the training pipeline, then load the saved merged_phase1_phase2_bundle.joblib here."
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.imputer_num is None or self.scaler is None:
            raise RuntimeError("Phase2FactorFeatureBuilder must be fitted before transform().")

        X_raw = self._make_phase2_raw(df)
        self._validate_required_factor_columns(X_raw)
        feature_cols = [col for col in X_raw.columns if col != ID_COL]
        X_features = X_raw[feature_cols].copy()

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


def register_joblib_compatibility() -> None:
    """Register class aliases so bundles saved from notebooks/scripts can load."""
    main_module = sys.modules.get("__main__")
    if main_module is not None:
        setattr(main_module, "Phase2FactorFeatureBuilder", Phase2FactorFeatureBuilder)

    module_name = "merged_phase1_phase2_pipeline"
    if module_name not in sys.modules:
        alias_module = types.ModuleType(module_name)
        alias_module.Phase2FactorFeatureBuilder = Phase2FactorFeatureBuilder
        alias_module.FA_SPECS = FA_SPECS
        alias_module.OWNER_HOUSEHOLD_VARS = OWNER_HOUSEHOLD_VARS
        alias_module.DROP_FROM_PHASE2 = DROP_FROM_PHASE2
        alias_module.ID_COL = ID_COL
        sys.modules[module_name] = alias_module


# -----------------------------------------------------------------------------
# Dashboard helper functions
# -----------------------------------------------------------------------------

def parse_optional_float(value: str) -> float:
    value = str(value).strip()
    if value == "":
        return np.nan
    return float(value)


YEAR_LABELS = {
    "2020": "four years before",
    "2021": "three years before",
    "2022": "two years before",
    "2023": "one year before",
}


def clean_display_name(name: str) -> str:
    """Convert training column names into readable dashboard labels."""
    replacements = {
        "foraml": "formal",
        "MSME": "MSME",
    }
    cleaned = name.replace("_", " ").replace("-", " ")
    for old_text, new_text in replacements.items():
        cleaned = cleaned.replace(old_text, new_text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Keep simple title casing for readability.
    words = []
    for word in cleaned.split(" "):
        if word.upper() == "MSME":
            words.append("MSME")
        else:
            words.append(word.capitalize())
    return " ".join(words)


def friendly_feature_label(column_name: str) -> str:
    """Show a user-friendly label while keeping the original model column name internally."""
    for year, period_label in YEAR_LABELS.items():
        suffix = f"_{year}"
        middle = f"_{year}_"

        if column_name.endswith(suffix):
            base_name = column_name[: -len(suffix)]
            return f"{clean_display_name(base_name)} ({period_label})"

        if middle in column_name:
            base_name = column_name.replace(middle, "_")
            return f"{clean_display_name(base_name)} ({period_label})"

    return clean_display_name(column_name)


def optional_numeric_input(label: str, key: str, help_text: str = "") -> float:
    raw = st.text_input(label, value="", key=key, help=help_text)
    try:
        return parse_optional_float(raw)
    except ValueError:
        st.error(f"{label}: enter a number or leave it blank.")
        st.stop()


def binary_input(label: str, key: str, feature_name: str, help_text: str = "") -> float:
    if feature_name == "Owner_gender":
        # Keep the numeric code visible so it matches the training data.
        # Swap these labels if your training dataset used the opposite gender coding.
        options = ["Missing", "0 - Female", "1 - Male"]
    else:
        options = ["Missing", "0 - No", "1 - Yes"]

    choice = st.selectbox(label, options=options, index=0, key=key, help=help_text)
    if choice == "Missing":
        return np.nan
    return float(choice.split(" - ")[0])


@st.cache_resource
def load_bundle(bundle_path: str | Path) -> Dict[str, Any]:
    register_joblib_compatibility()
    return joblib.load(bundle_path)


def validate_phase2_bundle(bundle: Dict[str, Any]) -> None:
    required_keys = ["phase2_feature_builder", "phase2_model", "phase2_feature_columns"]
    missing = [key for key in required_keys if key not in bundle]
    if missing:
        raise ValueError(
            "This does not look like the merged Phase 1 -> Phase 2 bundle. "
            f"Missing key(s): {missing}. Use merged_phase1_phase2_bundle.joblib, not the old Phase 1-only bundle."
        )


def get_builder(bundle: Dict[str, Any]) -> Phase2FactorFeatureBuilder:
    return bundle["phase2_feature_builder"]


def required_dashboard_columns(builder: Phase2FactorFeatureBuilder) -> List[str]:
    cols = [ID_COL]
    cols.extend(builder.numeric_columns)
    cols.extend(builder.categorical_columns)
    return list(dict.fromkeys(cols))


def add_missing_columns(df: pd.DataFrame, columns: List[str]) -> tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    out.columns = out.columns.str.strip()
    missing = [col for col in columns if col not in out.columns]
    for col in missing:
        out[col] = np.nan
    return out.reindex(columns=columns), missing


def risk_label(p_survival: float) -> str:
    if p_survival >= 0.70:
        return "Lower risk / likely active"
    if p_survival >= 0.40:
        return "Moderate risk"
    return "Higher risk / low survival probability"


def predict_phase2_survival(bundle: Dict[str, Any], raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    builder = get_builder(bundle)
    model = bundle["phase2_model"]
    expected_features = bundle["phase2_feature_columns"]

    X2 = builder.transform(raw_df)
    X2 = X2.reindex(columns=expected_features)

    survival_prob = np.clip(model.predict(X2), 0, 1)
    result = pd.DataFrame(index=raw_df.index)
    result[ID_COL] = raw_df[ID_COL].values if ID_COL in raw_df.columns else np.arange(1, len(raw_df) + 1)
    result["survival_probability"] = survival_prob
    result["failure_probability"] = 1.0 - survival_prob
    result["risk_label"] = [risk_label(float(p)) for p in survival_prob]
    return result, X2


def render_prediction_cards(result_row: pd.Series, model_name: str) -> None:
    p_survival = float(result_row["survival_probability"])
    p_failure = float(result_row["failure_probability"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated survival probability", f"{p_survival * 100:.2f}%")
    c2.metric("Estimated failure probability", f"{p_failure * 100:.2f}%")
    c3.metric("Phase 2 model", model_name)

    label = risk_label(p_survival)
    if p_survival >= 0.70:
        st.success(label)
    elif p_survival >= 0.40:
        st.warning(label)
    else:
        st.error(label)


def render_numeric_group(group_label: str, columns: List[str], raw_values: Dict[str, float]) -> None:
    with st.expander(group_label, expanded=(group_label == "Owner / household")):
        cols = st.columns(3)
        for i, col_name in enumerate(columns):
            with cols[i % 3]:
                raw_values[col_name] = optional_numeric_input(
                    label=friendly_feature_label(col_name),
                    key=f"num_{col_name}",
                    help_text=f"Model column: {col_name}. Leave blank if unknown. Missing values are handled using the training-set imputer.",
                )


def collect_manual_input(builder: Phase2FactorFeatureBuilder) -> tuple[pd.DataFrame, bool]:
    numeric_columns = list(builder.numeric_columns)
    categorical_columns = list(builder.categorical_columns)

    raw_values: Dict[str, Any] = {}
    submitted = False

    with st.form("phase2_manual_input"):
        st.subheader("Manual Phase 2 input")
        st.caption(
            "Enter the available values. Blank numeric values are allowed and will be imputed using the fitted training pipeline. "
            "For binary columns, use 0 - No / 1 - Yes. For owner gender, use Female or Male according to the training data coding."
        )

        raw_values[ID_COL] = st.text_input("Business ID", value="new_business", key="business_id")

        already_rendered = set()
        for group_label, group_cols in INPUT_GROUPS.items():
            use_cols = [c for c in group_cols if c in numeric_columns]
            already_rendered.update(use_cols)
            if use_cols:
                render_numeric_group(group_label, use_cols, raw_values)

        other_numeric_cols = [c for c in numeric_columns if c not in already_rendered]
        if other_numeric_cols:
            render_numeric_group("Other continuous variables", other_numeric_cols, raw_values)

        if categorical_columns:
            with st.expander("Categorical / binary variables", expanded=False):
                cols = st.columns(3)
                for i, col_name in enumerate(categorical_columns):
                    with cols[i % 3]:
                        raw_values[col_name] = binary_input(
                            label=friendly_feature_label(col_name),
                            key=f"cat_{col_name}",
                            feature_name=col_name,
                            help_text="Leave as Missing if unknown. The selected value is converted back to the original 0/1 code used by the model.",
                        )

        submitted = st.form_submit_button("Predict survival probability")

    raw_df = pd.DataFrame([raw_values])
    needed_cols = required_dashboard_columns(builder)
    raw_df, _ = add_missing_columns(raw_df, needed_cols)
    return raw_df, submitted


def read_uploaded_input(uploaded_file, builder: Phase2FactorFeatureBuilder) -> tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(uploaded_file)
    if ID_COL not in df.columns:
        df.insert(0, ID_COL, [f"business_{i + 1}" for i in range(len(df))])
    needed_cols = required_dashboard_columns(builder)
    return add_missing_columns(df, needed_cols)


def show_lightgbm_importance(bundle: Dict[str, Any]) -> None:
    model_pipeline = bundle.get("phase2_model")
    feature_cols = bundle.get("phase2_feature_columns", [])

    model = model_pipeline
    if hasattr(model_pipeline, "named_steps") and "model" in model_pipeline.named_steps:
        model = model_pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        st.info("The selected Phase 2 model does not provide built-in feature_importances_. Use permutation importance in the notebook instead.")
        return

    importance = np.asarray(model.feature_importances_, dtype=float)
    if len(importance) != len(feature_cols):
        st.info("Feature importance length does not match the Phase 2 feature columns.")
        return

    imp_df = pd.DataFrame({"feature": feature_cols, "importance": importance})
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
    st.dataframe(imp_df.head(30), use_container_width=True)
    st.bar_chart(imp_df.head(20).set_index("feature")["importance"])


def main() -> None:
    st.set_page_config(page_title="iMEWS+", layout="wide")
    st.title("Intelligent MSME Early Warning System")
    st.caption(
        "This dashboard predicts survival probability of a Micro Enterprise."
    )

    default_bundle_path = Path("model_bundle.joblib")
    bundle_path = st.sidebar.text_input("Merged model bundle path", str(default_bundle_path))

    try:
        bundle = load_bundle(bundle_path)
        validate_phase2_bundle(bundle)
    except FileNotFoundError:
        st.error("Bundle not found. Point the path to merged_phase1_phase2_bundle.joblib.")
        st.stop()
    except Exception as exc:
        st.error(f"Could not load the merged Phase 2 bundle: {exc}")
        st.stop()

    builder = get_builder(bundle)
    model_name = bundle.get("phase2_best_model_name", "Unknown")

    st.sidebar.success(f"Loaded Phase 2 model: {model_name}")
    st.sidebar.write(f"Phase 2 raw numeric columns: {len(builder.numeric_columns)}")
    st.sidebar.write(f"Phase 2 categorical/binary columns: {len(builder.categorical_columns)}")

    mode = st.radio(
        "Input method",
        options=["Manual input", "Upload CSV"],
        horizontal=True,
    )

    raw_df: Optional[pd.DataFrame] = None
    submitted = False
    missing_added: List[str] = []

    if mode == "Manual input":
        raw_df, submitted = collect_manual_input(builder)
    else:
        uploaded = st.file_uploader("Upload a CSV with one or more businesses", type=["csv"])
        st.caption("The CSV should use the same column names as the training dataset. Missing columns will be added as blank values and imputed.")
        if uploaded is not None:
            raw_df, missing_added = read_uploaded_input(uploaded, builder)
            submitted = st.button("Predict survival probability for uploaded rows")
            if missing_added:
                with st.expander("Columns added as missing values"):
                    st.write(missing_added)
        else:
            st.info("Upload a CSV file to continue.")

    if raw_df is None:
        return

    tab_pred, tab_input, tab_features, tab_importance = st.tabs(
        ["Prediction", "Raw input", "Computed Phase 2 features", "Model feature importance"]
    )

    result_df: Optional[pd.DataFrame] = None
    X2: Optional[pd.DataFrame] = None

    if submitted:
        try:
            result_df, X2 = predict_phase2_survival(bundle, raw_df)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

    with tab_pred:
        if submitted and result_df is not None:
            if len(result_df) == 1:
                render_prediction_cards(result_df.iloc[0], model_name)
            st.subheader("Prediction output")
            display_df = result_df.copy()
            display_df["survival_probability_percent"] = display_df["survival_probability"] * 100
            display_df["failure_probability_percent"] = display_df["failure_probability"] * 100
            st.dataframe(display_df, use_container_width=True)

            csv_bytes = display_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download predictions as CSV",
                data=csv_bytes,
                file_name="phase2_dashboard_predictions.csv",
                mime="text/csv",
            )

            st.info(
                "This probability is the Phase 2 estimate of the Phase 1 survival probability. "
                "It is not a direct Business_Status classifier output."
            )
        else:
            st.info("Enter values or upload a CSV, then click the prediction button.")

    with tab_input:
        st.subheader("Raw values sent into the fitted Phase 2 feature builder")
        st.dataframe(raw_df, use_container_width=True)

    with tab_features:
        if submitted and X2 is not None:
            st.subheader("Computed factor scores, scaled direct variables, and categorical variables")
            if len(X2) == 1:
                st.dataframe(X2.T.rename(columns={X2.index[0]: "value"}), use_container_width=True)
            else:
                st.dataframe(X2, use_container_width=True)
        else:
            st.info("Computed Phase 2 features will appear after prediction.")

    with tab_importance:
        st.subheader("Built-in model feature importance")
        show_lightgbm_importance(bundle)
        st.caption(
            "For final reporting, permutation importance is usually better because it measures how prediction error changes when a feature is shuffled. "
            "Built-in LightGBM importance is still useful as a quick check."
        )


if __name__ == "__main__":
    main()
