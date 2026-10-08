"""Dataset profiling module for AI Analytics Engineer.

Performs deterministic statistical analysis, column classification,
correlation computation, outlier detection, and insight generation using Pandas.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def detect_column_type(series: pd.Series) -> str:
    """Detect high-level logical data type for a pandas Series."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numerical"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    
    # Check if object series can be safely parsed as datetime
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        non_null = series.dropna()
        if len(non_null) > 0 and len(non_null) <= 500:
            try:
                # Try parsing sample to see if it's dates
                sample = non_null.head(20)
                parsed = pd.to_datetime(sample, format="mixed", errors="coerce")
                if parsed.notna().all() and parsed.nunique() > 1:
                    return "datetime"
            except Exception:
                pass
        return "categorical"
    
    return "other"


def profile_general(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate general dataset-level statistics."""
    total_rows = int(len(df))
    total_columns = int(len(df.columns))
    duplicate_rows = int(df.duplicated().sum()) if total_rows > 0 else 0
    duplicate_pct = (duplicate_rows / total_rows * 100) if total_rows > 0 else 0.0
    memory_mb = float(df.memory_usage(deep=True).sum() / (1024 * 1024))
    
    return {
        "total_rows": total_rows,
        "total_columns": total_columns,
        "duplicate_rows": duplicate_rows,
        "duplicate_percentage": round(duplicate_pct, 2),
        "memory_usage_mb": round(memory_mb, 2),
        "total_cells": total_rows * total_columns,
        "total_missing_cells": int(df.isna().sum().sum()),
        "total_missing_percentage": round((df.isna().sum().sum() / (total_rows * total_columns) * 100), 2) if (total_rows * total_columns) > 0 else 0.0,
    }


def profile_columns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Profile each column's metadata, missing values, and data type."""
    columns_info = []
    total_rows = len(df)
    
    for col in df.columns:
        series = df[col]
        col_type = detect_column_type(series)
        missing_count = int(series.isna().sum())
        missing_pct = round((missing_count / total_rows * 100), 2) if total_rows > 0 else 0.0
        unique_count = int(series.nunique(dropna=True))
        
        # Sample non-null values
        sample_vals = [str(x) for x in series.dropna().unique()[:5].tolist()]
        
        columns_info.append({
            "name": str(col),
            "logical_type": col_type,
            "raw_dtype": str(series.dtype),
            "unique_count": unique_count,
            "missing_count": missing_count,
            "missing_percentage": missing_pct,
            "sample_values": sample_vals,
        })
        
    return columns_info


def profile_numerical(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Compute detailed descriptive statistics for numerical columns."""
    numerical_stats = {}
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
            
        q25 = float(series.quantile(0.25))
        q50 = float(series.quantile(0.50))
        q75 = float(series.quantile(0.75))
        iqr = q75 - q25
        
        numerical_stats[str(col)] = {
            "count": int(len(series)),
            "mean": round(float(series.mean()), 4),
            "median": round(float(series.median()), 4),
            "std": round(float(series.std()), 4) if len(series) > 1 else 0.0,
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "q25": round(q25, 4),
            "q50": round(q50, 4),
            "q75": round(q75, 4),
            "iqr": round(iqr, 4),
            "sum": round(float(series.sum()), 4),
        }
        
    return numerical_stats


def profile_categorical(df: pd.DataFrame, top_k: int = 5) -> Dict[str, Dict[str, Any]]:
    """Analyze categorical columns for cardinality and top frequent items."""
    categorical_stats = {}
    
    for col in df.columns:
        if detect_column_type(df[col]) in ("categorical", "boolean", "other"):
            series = df[col].dropna().astype(str)
            if len(series) == 0:
                continue
                
            val_counts = series.value_counts()
            unique_count = int(len(val_counts))
            
            top_values = []
            for val, count in val_counts.head(top_k).items():
                pct = round((count / len(series) * 100), 2)
                top_values.append({
                    "value": str(val),
                    "count": int(count),
                    "percentage": pct
                })
                
            categorical_stats[str(col)] = {
                "unique_count": unique_count,
                "total_non_null": int(len(series)),
                "top_values": top_values,
                "mode": str(val_counts.index[0]) if len(val_counts) > 0 else None,
                "mode_frequency": int(val_counts.iloc[0]) if len(val_counts) > 0 else 0,
            }
            
    return categorical_stats


def compute_correlations(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Calculate Pearson correlation matrix for numerical columns."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2 or len(num_df) < 2:
        return None
        
    corr = num_df.corr(method="pearson").round(3)
    return corr


def detect_outliers_iqr(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Detect outliers in numerical columns using the explainable IQR method."""
    outliers_info = {}
    num_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in num_cols:
        series = df[col].dropna()
        if len(series) < 4:
            continue
            
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_count = int(outlier_mask.sum())
        outlier_pct = round((outlier_count / len(series) * 100), 2) if len(series) > 0 else 0.0
        
        outliers_info[str(col)] = {
            "q1": round(q1, 3),
            "q3": round(q3, 3),
            "iqr": round(iqr, 3),
            "lower_bound": round(lower_bound, 3),
            "upper_bound": round(upper_bound, 3),
            "outlier_count": outlier_count,
            "outlier_percentage": outlier_pct,
            "has_outliers": outlier_count > 0,
        }
        
    return outliers_info


def generate_automatic_insights(
    df: pd.DataFrame,
    general: Dict[str, Any],
    columns: List[Dict[str, Any]],
    numerical_stats: Dict[str, Dict[str, Any]],
    categorical_stats: Dict[str, Dict[str, Any]],
    correlations: Optional[pd.DataFrame],
    outliers: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate deterministic, data-backed insights from the dataset."""
    insights = []
    
    # 1. Dataset Dimensions & Completeness
    if general["total_rows"] > 0:
        if general["total_missing_percentage"] == 0:
            insights.append({
                "type": "data_quality",
                "category": "Quality",
                "title": "Clean Dataset",
                "description": f"Dataset contains {general['total_rows']:,} rows and {general['total_columns']} columns with 0% missing values.",
                "importance": "info"
            })
        elif general["total_missing_percentage"] > 10:
            insights.append({
                "type": "data_quality",
                "category": "Quality",
                "title": "High Missing Data",
                "description": f"Overall missing values rate is {general['total_missing_percentage']}% across {general['total_missing_cells']:,} cells.",
                "importance": "warning"
            })
            
    # 2. Duplicate rows
    if general["duplicate_rows"] > 0:
        insights.append({
            "type": "duplicates",
            "category": "Quality",
            "title": "Duplicate Rows Detected",
            "description": f"Found {general['duplicate_rows']:,} duplicate rows ({general['duplicate_percentage']}% of the dataset).",
            "importance": "warning"
        })

    # 3. Columns with significant missing data
    high_missing_cols = [c for c in columns if c["missing_percentage"] >= 5.0]
    if high_missing_cols:
        col_names = ", ".join([f"{c['name']} ({c['missing_percentage']}%)" for c in high_missing_cols[:3]])
        insights.append({
            "type": "missing_columns",
            "category": "Quality",
            "title": "Columns with High Missingness",
            "description": f"Columns requiring attention: {col_names}.",
            "importance": "warning"
        })

    # 4. Highest Average Numerical Metric
    if numerical_stats:
        highest_mean_col = max(numerical_stats.items(), key=lambda x: x[1]["mean"])
        insights.append({
            "type": "numerical_highest_mean",
            "category": "Metrics",
            "title": f"Highest Mean: {highest_mean_col[0]}",
            "description": f"Column '{highest_mean_col[0]}' has the highest average value at {highest_mean_col[1]['mean']:,.2f} (median: {highest_mean_col[1]['median']:,.2f}).",
            "importance": "info"
        })
        
        # Highest Total Sum
        highest_sum_col = max(numerical_stats.items(), key=lambda x: x[1]["sum"])
        if highest_sum_col[0] != highest_mean_col[0]:
            insights.append({
                "type": "numerical_highest_sum",
                "category": "Metrics",
                "title": f"Highest Total: {highest_sum_col[0]}",
                "description": f"Column '{highest_sum_col[0]}' accounts for the largest cumulative sum at {highest_sum_col[1]['sum']:,.2f}.",
                "importance": "info"
            })

    # 5. Dominant Categorical Values
    if categorical_stats:
        for col_name, cat_data in categorical_stats.items():
            if cat_data["top_values"] and cat_data["top_values"][0]["percentage"] >= 35.0:
                top_val = cat_data["top_values"][0]
                insights.append({
                    "type": "dominant_category",
                    "category": "Distribution",
                    "title": f"Dominant '{col_name}': {top_val['value']}",
                    "description": f"'{top_val['value']}' represents {top_val['percentage']}% ({top_val['count']:,} occurrences) of all '{col_name}' records.",
                    "importance": "info"
                })
                break

    # 6. Strong Correlations (|r| >= 0.6)
    if correlations is not None and not correlations.empty:
        seen_pairs = set()
        strong_pairs = []
        
        for col1 in correlations.columns:
            for col2 in correlations.columns:
                if col1 != col2:
                    pair = tuple(sorted([col1, col2]))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        r = correlations.loc[col1, col2]
                        if not np.isnan(r) and abs(r) >= 0.6:
                            strong_pairs.append((col1, col2, r))
                            
        strong_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for c1, c2, r in strong_pairs[:3]:
            direction = "positive" if r > 0 else "inverse (negative)"
            insights.append({
                "type": "strong_correlation",
                "category": "Correlation",
                "title": f"Strong Correlation: {c1} & {c2}",
                "description": f"Strong {direction} correlation (r = {r:+.2f}) detected between '{c1}' and '{c2}'.",
                "importance": "high"
            })

    # 7. Outliers
    if outliers:
        cols_with_outliers = [(col, data) for col, data in outliers.items() if data["has_outliers"]]
        if cols_with_outliers:
            cols_with_outliers.sort(key=lambda x: x[1]["outlier_count"], reverse=True)
            top_outlier = cols_with_outliers[0]
            insights.append({
                "type": "outlier_alert",
                "category": "Anomalies",
                "title": f"Outliers in '{top_outlier[0]}'",
                "description": f"Detected {top_outlier[1]['outlier_count']} outlier rows ({top_outlier[1]['outlier_percentage']}%) outside normal IQR bounds [{top_outlier[1]['lower_bound']:,.2f}, {top_outlier[1]['upper_bound']:,.2f}].",
                "importance": "warning"
            })

    return insights


def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Complete profile of the dataset combining all analytical dimensions."""
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a valid pandas DataFrame.")
        
    general = profile_general(df)
    columns = profile_columns(df)
    numerical = profile_numerical(df)
    categorical = profile_categorical(df)
    correlations = compute_correlations(df)
    outliers = detect_outliers_iqr(df)
    
    insights = generate_automatic_insights(
        df=df,
        general=general,
        columns=columns,
        numerical_stats=numerical,
        categorical_stats=categorical,
        correlations=correlations,
        outliers=outliers,
    )
    
    return {
        "general": general,
        "columns": columns,
        "numerical": numerical,
        "categorical": categorical,
        "correlations": correlations,
        "outliers": outliers,
        "insights": insights,
    }
