"""Unit tests for src/analyzer.py."""

import numpy as np
import pandas as pd
import pytest
from src.analyzer import AnalyticsEngine, resolve_column_name


@pytest.fixture
def test_df():
    """Create a deterministic test dataset for analysis."""
    return pd.DataFrame({
        "product": ["Widget A", "Widget B", "Widget C", "Widget A", "Widget B", "Widget C"],
        "category": ["Electronics", "Electronics", "Home", "Electronics", "Electronics", "Home"],
        "region": ["North", "North", "South", "South", "North", "South"],
        "revenue": [100.0, 200.0, 300.0, 150.0, 250.0, 350.0],
        "units": [10, 20, 30, 15, 25, 35],
        "rating": [4.5, 4.0, 3.5, 4.8, 4.2, 3.8],
        "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-02-01", "2025-02-02", "2025-03-01", "2025-03-02"]),
    })


def test_resolve_column_name(test_df):
    """Test flexible column name resolution."""
    assert resolve_column_name("revenue", test_df) == "revenue"
    assert resolve_column_name("Revenue", test_df) == "revenue"
    assert resolve_column_name("REVENUE", test_df) == "revenue"
    assert resolve_column_name("  product ", test_df) == "product"
    assert resolve_column_name("non_existent", test_df) is None


def test_aggregate_mean_and_sum(test_df):
    """Test aggregation operations."""
    engine = AnalyticsEngine(test_df)
    
    # Mean
    res_mean = engine.execute({"operation": "aggregate", "column": "revenue", "aggregation": "mean"})
    assert res_mean.success is True
    assert res_mean.data == round(float(test_df["revenue"].mean()), 4)
    assert "MEAN(revenue)" in res_mean.formatted_result
    
    # Sum
    res_sum = engine.execute({"operation": "aggregate", "column": "units", "aggregation": "sum"})
    assert res_sum.success is True
    assert res_sum.data == float(test_df["units"].sum())


def test_groupby_aggregate(test_df):
    """Test groupby operations with sorting and limits."""
    engine = AnalyticsEngine(test_df)
    
    # Group by category, sum revenue, descending
    res = engine.execute({
        "operation": "groupby_aggregate",
        "group_column": "category",
        "value_column": "revenue",
        "aggregation": "sum",
        "sort": "descending"
    })
    
    assert res.success is True
    assert isinstance(res.summary_table, pd.DataFrame)
    assert len(res.summary_table) == 2
    # Electronics total: 100+200+150+250 = 700. Home total: 300+350 = 650
    assert res.summary_table.iloc[0]["category"] == "Electronics"
    assert res.summary_table.iloc[0]["sum_revenue"] == 700.0


def test_top_n(test_df):
    """Test top N grouping."""
    engine = AnalyticsEngine(test_df)
    
    res = engine.execute({
        "operation": "top_n",
        "group_column": "product",
        "column": "revenue",
        "n": 2,
        "aggregation": "sum"
    })
    
    assert res.success is True
    assert len(res.summary_table) == 2
    # Widget C total = 650, Widget B total = 450, Widget A total = 250
    assert res.summary_table.iloc[0]["product"] == "Widget C"
    assert res.summary_table.iloc[0]["sum_revenue"] == 650.0


def test_filter_operation(test_df):
    """Test filtering conditions."""
    engine = AnalyticsEngine(test_df)
    
    res = engine.execute({
        "operation": "filter",
        "conditions": [{"column": "region", "operator": "==", "value": "North"}],
        "select_columns": ["product", "revenue"]
    })
    
    assert res.success is True
    assert len(res.summary_table) == 3
    assert set(res.summary_table.columns) == {"product", "revenue"}


def test_correlation(test_df):
    """Test correlation computation between numerical columns."""
    engine = AnalyticsEngine(test_df)
    
    res = engine.execute({
        "operation": "correlation",
        "column1": "revenue",
        "column2": "units"
    })
    
    assert res.success is True
    assert res.data == 1.0  # Perfect correlation in test data


def test_time_series(test_df):
    """Test time-series aggregation."""
    engine = AnalyticsEngine(test_df)
    
    res = engine.execute({
        "operation": "time_series",
        "date_column": "date",
        "value_column": "revenue",
        "time_granularity": "M",
        "aggregation": "sum"
    })
    
    assert res.success is True
    assert len(res.summary_table) == 3  # Jan, Feb, Mar


def test_invalid_column_error(test_df):
    """Test error handling when querying non-existent column."""
    engine = AnalyticsEngine(test_df)
    
    res = engine.execute({
        "operation": "aggregate",
        "column": "invalid_column_xyz",
        "aggregation": "mean"
    })
    
    assert res.success is False
    assert "not found" in res.error.lower()


def test_non_numerical_aggregation_error(test_df):
    """Test error handling when trying to calculate mean of string column."""
    engine = AnalyticsEngine(test_df)
    
    res = engine.execute({
        "operation": "aggregate",
        "column": "product",
        "aggregation": "mean"
    })
    
    assert res.success is False
    assert "not numerical" in res.error.lower()
