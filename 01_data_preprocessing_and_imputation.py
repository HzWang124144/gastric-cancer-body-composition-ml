#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
01_data_preprocessing_and_imputation.py

This script performs data preprocessing and missing-value imputation for the
gastric cancer body composition machine-learning study.

Main functions:
1. Load institutional tabular data from CSV or Excel.
2. Summarize missingness for each candidate predictor.
3. Exclude outcome variables from imputation.
4. Impute numeric predictors using scikit-learn IterativeImputer.
5. Impute categorical predictors using the most frequent category.
6. Export imputed data, missingness summary, and imputation audit tables.

Notes:
- Individual-level clinical data are not included in this repository.
- Users should replace the input path with their own authorized dataset.
- Outcome variables should not be imputed.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_table(input_path: str) -> pd.DataFrame:
    """
    Load a CSV or Excel file.

    Parameters
    ----------
    input_path : str
        Path to the input dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() in [".xlsx", ".xls"]:
        data = pd.read_excel(input_path)
    elif input_path.suffix.lower() == ".csv":
        data = pd.read_csv(input_path)
    else:
        raise ValueError("Unsupported file format. Please use .csv, .xlsx, or .xls.")

    return data


# ---------------------------------------------------------------------
# Missingness summary
# ---------------------------------------------------------------------

def summarize_missingness(data: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize missing values for each variable.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.

    Returns
    -------
    pd.DataFrame
        Missingness summary table.
    """
    n_total = len(data)

    summary = pd.DataFrame({
        "Variable": data.columns,
        "Missing_n": data.isna().sum().values,
        "Missing_percent": data.isna().mean().values * 100,
        "Data_type": [str(data[col].dtype) for col in data.columns]
    })

    summary["Total_n"] = n_total
    summary = summary.sort_values(
        by=["Missing_n", "Variable"],
        ascending=[False, True]
    ).reset_index(drop=True)

    return summary


# ---------------------------------------------------------------------
# Column classification
# ---------------------------------------------------------------------

def split_columns_by_type(
    data: pd.DataFrame,
    exclude_columns: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split columns into numeric predictors, categorical predictors, and excluded columns.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    exclude_columns : list of str
        Columns excluded from imputation, such as outcome variables or IDs.

    Returns
    -------
    numeric_cols : list of str
        Numeric predictor columns to be imputed with IterativeImputer.
    categorical_cols : list of str
        Categorical predictor columns to be imputed with SimpleImputer.
    excluded_existing : list of str
        Excluded columns found in the dataset.
    """
    excluded_existing = [col for col in exclude_columns if col in data.columns]
    candidate_cols = [col for col in data.columns if col not in excluded_existing]

    numeric_cols = data[candidate_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in candidate_cols if col not in numeric_cols]

    return numeric_cols, categorical_cols, excluded_existing


# ---------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------

def impute_data(
    data: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    random_state: int = 42,
    max_iter: int = 10
) -> pd.DataFrame:
    """
    Impute missing values in numeric and categorical predictors.

    Numeric variables are imputed using IterativeImputer.
    Categorical variables are imputed using the most frequent category.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    numeric_cols : list of str
        Numeric columns for IterativeImputer.
    categorical_cols : list of str
        Categorical columns for SimpleImputer.
    random_state : int
        Random seed for reproducibility.
    max_iter : int
        Maximum number of imputation iterations.

    Returns
    -------
    pd.DataFrame
        Dataset after imputation.
    """
    imputed_data = data.copy()

    # Numeric imputation
    if numeric_cols:
        valid_numeric_cols = [
            col for col in numeric_cols
            if not imputed_data[col].isna().all()
        ]

        all_missing_numeric_cols = [
            col for col in numeric_cols
            if imputed_data[col].isna().all()
        ]

        if all_missing_numeric_cols:
            print(
                "Warning: The following numeric columns are entirely missing "
                "and will not be imputed:"
            )
            print(all_missing_numeric_cols)

        if valid_numeric_cols:
            numeric_imputer = IterativeImputer(
                random_state=random_state,
                max_iter=max_iter,
                sample_posterior=False
            )

            imputed_numeric = numeric_imputer.fit_transform(
                imputed_data[valid_numeric_cols]
            )

            imputed_data[valid_numeric_cols] = pd.DataFrame(
                imputed_numeric,
                columns=valid_numeric_cols,
                index=imputed_data.index
            )

    # Categorical imputation
    if categorical_cols:
        valid_categorical_cols = [
            col for col in categorical_cols
            if not imputed_data[col].isna().all()
        ]

        all_missing_categorical_cols = [
            col for col in categorical_cols
            if imputed_data[col].isna().all()
        ]

        if all_missing_categorical_cols:
            print(
                "Warning: The following categorical columns are entirely missing "
                "and will not be imputed:"
            )
            print(all_missing_categorical_cols)

        if valid_categorical_cols:
            categorical_imputer = SimpleImputer(strategy="most_frequent")

            imputed_categorical = categorical_imputer.fit_transform(
                imputed_data[valid_categorical_cols]
            )

            imputed_data[valid_categorical_cols] = pd.DataFrame(
                imputed_categorical,
                columns=valid_categorical_cols,
                index=imputed_data.index
            )

    return imputed_data


# ---------------------------------------------------------------------
# Imputation audit table
# ---------------------------------------------------------------------

def create_imputation_audit_table(
    original_data: pd.DataFrame,
    imputed_data: pd.DataFrame,
    imputed_columns: List[str]
) -> pd.DataFrame:
    """
    Create an audit table recording the location and value of each imputed cell.

    Parameters
    ----------
    original_data : pd.DataFrame
        Dataset before imputation.
    imputed_data : pd.DataFrame
        Dataset after imputation.
    imputed_columns : list of str
        Columns included in imputation.

    Returns
    -------
    pd.DataFrame
        Imputation audit table.
    """
    records = []

    for row_idx in range(original_data.shape[0]):
        for col in imputed_columns:
            if col in original_data.columns and pd.isna(original_data.loc[row_idx, col]):
                records.append({
                    "DataFrame_row": row_idx,
                    "Excel_row_if_header_in_row_1": row_idx + 2,
                    "Variable": col,
                    "Original_value": original_data.loc[row_idx, col],
                    "Imputed_value": imputed_data.loc[row_idx, col]
                })

    audit_table = pd.DataFrame(records)

    return audit_table


def summarize_imputed_values(audit_table: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize the number of imputed values by variable.

    Parameters
    ----------
    audit_table : pd.DataFrame
        Imputation audit table.

    Returns
    -------
    pd.DataFrame
        Summary table of imputed values by variable.
    """
    if audit_table.empty:
        return pd.DataFrame(columns=["Variable", "Imputed_n"])

    summary = (
        audit_table
        .groupby("Variable")
        .size()
        .reset_index(name="Imputed_n")
        .sort_values("Imputed_n", ascending=False)
        .reset_index(drop=True)
    )

    return summary


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def save_outputs(
    original_data: pd.DataFrame,
    imputed_data: pd.DataFrame,
    missingness_summary: pd.DataFrame,
    audit_table: pd.DataFrame,
    audit_summary: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Save preprocessing and imputation outputs.

    Parameters
    ----------
    original_data : pd.DataFrame
        Original dataset.
    imputed_data : pd.DataFrame
        Imputed dataset.
    missingness_summary : pd.DataFrame
        Missingness summary.
    audit_table : pd.DataFrame
        Imputation audit table.
    audit_summary : pd.DataFrame
        Summary of imputed values by variable.
    output_dir : str
        Output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    missingness_summary.to_csv(
        output_dir / "missingness_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    imputed_data.to_csv(
        output_dir / "imputed_data.csv",
        index=False,
        encoding="utf-8-sig"
    )

    audit_table.to_csv(
        output_dir / "imputed_values_audit.csv",
        index=False,
        encoding="utf-8-sig"
    )

    audit_summary.to_csv(
        output_dir / "imputed_values_summary_by_variable.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Excel workbook for local checking
    excel_path = output_dir / "imputation_check.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        missingness_summary.to_excel(
            writer,
            sheet_name="Missingness_summary",
            index=False
        )
        audit_table.to_excel(
            writer,
            sheet_name="Imputed_values",
            index=False
        )
        audit_summary.to_excel(
            writer,
            sheet_name="Summary_by_variable",
            index=False
        )
        imputed_data.to_excel(
            writer,
            sheet_name="Imputed_data",
            index=False
        )

    print(f"Outputs saved to: {output_dir}")


# ---------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess data and perform missing-value imputation."
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV or Excel file."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/preprocessing",
        help="Directory for output files."
    )

    parser.add_argument(
        "--exclude-columns",
        type=str,
        nargs="*",
        default=[
            "patient_id",
            "recurrence",
            "recurrence_status",
            "three_year_survival",
            "3_year_survival",
            "survival_status"
        ],
        help=(
            "Columns excluded from imputation, such as patient ID "
            "or outcome variables."
        )
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for IterativeImputer."
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=10,
        help="Maximum number of IterativeImputer iterations."
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Data preprocessing and missing-value imputation")
    print("=" * 80)

    # Load data
    data = load_table(args.input)
    print(f"Loaded data shape: {data.shape}")

    # Summarize missingness before imputation
    missingness_summary = summarize_missingness(data)
    print("\nMissingness summary:")
    print(missingness_summary)

    # Split columns
    numeric_cols, categorical_cols, excluded_existing = split_columns_by_type(
        data=data,
        exclude_columns=args.exclude_columns
    )

    print("\nExcluded columns:")
    print(excluded_existing)

    print("\nNumeric columns for IterativeImputer:")
    print(numeric_cols)

    print("\nCategorical columns for most-frequent imputation:")
    print(categorical_cols)

    # Impute data
    imputed_data = impute_data(
        data=data,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        random_state=args.random_state,
        max_iter=args.max_iter
    )

    # Create audit tables
    imputed_columns = numeric_cols + categorical_cols
    audit_table = create_imputation_audit_table(
        original_data=data,
        imputed_data=imputed_data,
        imputed_columns=imputed_columns
    )

    audit_summary = summarize_imputed_values(audit_table)

    print("\nImputation audit summary:")
    print(audit_summary)

    # Save outputs
    save_outputs(
        original_data=data,
        imputed_data=imputed_data,
        missingness_summary=missingness_summary,
        audit_table=audit_table,
        audit_summary=audit_summary,
        output_dir=args.output_dir
    )

    print("\nPreprocessing and imputation completed.")


if __name__ == "__main__":
    main()
