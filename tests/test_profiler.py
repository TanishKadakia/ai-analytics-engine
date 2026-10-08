"""Unit tests for src/profiler.py."""

import numpy as np
import pandas as pd
import pytest
from src.profiler import (
    compute_correlations,
    detect_column_type,
    detect_outliers_iqr,
    generate_automatic_insights,
    profile_categorical,
    profile_columns,
    profile_dataset,
    profile_general,
    profile_numerical,
)


@pytest.fixture
def sample_df():
    """Create a deterministic test dataset."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "category": ["A", "B", "A", "B", "A", "C", "A", "B", "C", "A"],
        "revenue": [100.0, 200.0, 150.0, 300.0, 120.0, 400.0, 110.0, 250.0, 380.0, 1000.0],  # 1000 is an outlier
        "units": [10, 20, 15, 30, 12, 40, 11, 25, 38, 50],
        "rating": [4.5, 4.0, np.nan, 4.8, 3.9, 4.2, 4.7, np.nan, 4.1, 4.9],
        "is_active": [True, True, False, True, True, False, True, True, False, True],
        "date": pd.date_range("2025-01-01", periods=10, freq="D"),
    })


def test_profile_general(sample_df):
    """Test general metrics calculation."""
    general = profile_general(sample_df)
    assert general["total_rows"] == 10
    assert general["total_columns"] == 7
    assert general["duplicate_rows"] == 0
    assert general["duplicate_percentage"] == 0.0
    assert general["total_missing_cells"] == 2


def test_profile_columns(sample_df):
    """Test column metadata profiling."""
    columns = profile_columns(sample_df)
    assert len(columns) == 7
    
    col_map = {c["name"]: c for c in columns}
    assert col_map["revenue"]["logical_type"] == "numerical"
    assert col_map["category"]["logical_type"] == "categorical"
    assert col_map["rating"]["missing_count"] == 2
    assert col_map["rating"]["missing_percentage"] == 20.0
    assert col_map["category"]["unique_count"] == 3


def test_profile_numerical(sample_df):
    """Test descriptive statistics for numerical columns."""
    numerical = profile_numerical(sample_df)
    assert "revenue" in numerical
    assert "units" in numerical
    assert "rating" in numerical
    
    rev_stats = numerical["revenue"]
    assert rev_stats["count"] == 10
    assert rev_stats["min"] == 100.0
    assert rev_stats["max"] == 1000.0
    assert rev_stats["mean"] == 301.0
    assert rev_stats["median"] == 225.0


def test_profile_categorical(sample_df):
    """Test categorical value frequency analysis."""
    categorical = profile_categorical(sample_df)
    assert "category" in categorical
    cat_stats = categorical["category"]
    assert cat_stats["unique_count"] == 3
    assert cat_stats["mode"] == "A"
    assert cat_stats["mode_frequency"] == 5
    assert cat_stats["top_values"][0]["value"] == "A"
    assert cat_stats["top_values"][0]["percentage"] == 50.0


def test_correlations(sample_df):
    """Test Pearson correlation calculation."""
    corr = compute_correlations(sample_df)
    assert corr is not None
    assert "revenue" in corr.columns
    assert "units" in corr.columns
    # Revenue and units are positively correlated
    assert corr.loc["revenue", "units"] > 0.5


def test_detect_outliers_iqr(sample_df):
    """Test IQR outlier detection."""
    outliers = detect_outliers_iqr(sample_df)
    assert "revenue" in outliers
    assert outliers["revenue"]["has_outliers"] is True
    assert outliers["revenue"]["outlier_count"] >= 1
    assert outliers["revenue"]["upper_bound"] < 1000.0


def test_profile_dataset_complete(sample_df):
    """Test full dataset profiling workflow."""
    profile = profile_dataset(sample_df)
    assert "general" in profile
    assert "columns" in profile
    assert "numerical" in profile
    assert "categorical" in profile
    assert "correlations" in profile
    assert "outliers" in profile
    assert "insights" in profile
    assert len(profile["insights"]) > 0


def test_empty_and_edge_cases():
    """Test profiler robustness with edge cases."""
    empty_df = pd.DataFrame(columns=["a", "b"])
    general = profile_general(empty_df)
    assert general["total_rows"] == 0
    assert general["total_columns"] == 2
    
    # Categorical only
    cat_df = pd.DataFrame({"letter": ["A", "B", "C"], "word": ["foo", "bar", "baz"]})
    cat_profile = profile_dataset(cat_df)
    assert cat_profile["correlations"] is None
    assert len(cat_profile["numerical"]) == 0
