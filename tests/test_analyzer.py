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


def test_filter_rejects_unknown_column(test_df):
    """Invalid filter columns must not silently be ignored."""
    engine = AnalyticsEngine(test_df)

    res = engine.execute({
        "operation": "filter",
        "conditions": [
            {"column": "regoin", "operator": "==", "value": "North"}
        ]
    })

    assert res.success is False
    assert "column" in res.error.lower()


def test_filter_rejects_unknown_operator(test_df):
    """Unsupported operators must return a clear error."""
    engine = AnalyticsEngine(test_df)

    res = engine.execute({
        "operation": "filter",
        "conditions": [
            {"column": "region", "operator": "approximately", "value": "North"}
        ]
    })

    assert res.success is False
    assert "operator" in res.error.lower()


def test_filter_contains_treats_input_as_literal_text(test_df):
    """Regex metacharacters should be treated as literal search text."""
    df = test_df.copy()
    df["product"] = ["A.B", "AB", "A.B", "Widget", "A.B", "AB"]
    engine = AnalyticsEngine(df)

    res = engine.execute({
        "operation": "filter",
        "conditions": [
            {"column": "product", "operator": "contains", "value": "."}
        ]
    })

    assert res.success is True
    assert len(res.summary_table) == 3
    assert all(res.summary_table["product"] == "A.B")



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


def test_dataset_info_operation(test_df):
    """Test dataset_info operation returns exact rows and columns."""
    engine = AnalyticsEngine(test_df)
    res = engine.execute({"operation": "dataset_info"})

    assert res.success is True
    assert res.operation == "dataset_info"
    assert res.data["rows"] == len(test_df)
    assert res.data["columns"] == len(test_df.columns)
    assert f"{len(test_df)} rows" in res.formatted_result
    assert f"{len(test_df.columns)} columns" in res.formatted_result


def test_schema_operation(test_df):
    """Test schema operation returns all column names and dtypes."""
    engine = AnalyticsEngine(test_df)
    res = engine.execute({"operation": "schema"})

    assert res.success is True
    assert res.operation == "schema"
    assert isinstance(res.summary_table, pd.DataFrame)
    assert len(res.summary_table) == len(test_df.columns)

    col_names = res.summary_table["Column Name"].tolist()
    assert set(col_names) == set(test_df.columns)

    dtypes = dict(zip(res.summary_table["Column Name"], res.summary_table["Data Type"]))
    for col in test_df.columns:
        assert dtypes[col] == str(test_df[col].dtype)


def test_count_aggregation_no_decimals(test_df):
    """Test count aggregation is formatted without unnecessary decimal places."""
    engine = AnalyticsEngine(test_df)
    res = engine.execute({"operation": "aggregate", "column": "revenue", "aggregation": "count"})

    assert res.success is True
    assert res.data == len(test_df)
    assert res.formatted_result == f"COUNT(revenue) = {len(test_df)}"
    assert ".00" not in res.formatted_result


def test_top_n_multi_item_formatting(test_df):
    """Test top-N operation formats all ranked items in result."""
    engine = AnalyticsEngine(test_df)
    res = engine.execute({
        "operation": "groupby_aggregate",
        "group_column": "product",
        "value_column": "revenue",
        "aggregation": "sum",
        "sort": "descending",
        "limit": 3
    })

    assert res.success is True
    assert len(res.summary_table) == 3
    assert "Top 3 product by SUM(revenue):" in res.formatted_result
    assert "Widget C" in res.formatted_result
    assert "Widget B" in res.formatted_result
    assert "Widget A" in res.formatted_result
