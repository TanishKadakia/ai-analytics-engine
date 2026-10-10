"""End-to-end integration test for AI Analytics Engineer.

Tests full workflow: data loading, profiling, question parsing,
deterministic analysis, chart generation, and explanation.
"""

import os
from pathlib import Path
import pandas as pd
import pytest
import requests

from src.analyzer import AnalyticsEngine, AnalysisResult
from src.llm import explain_result_with_llm, parse_question_with_llm
from src.profiler import profile_dataset
from src.visualizer import create_chart_from_result


@pytest.fixture
def sample_dataset():
    """Load sample dataset."""
    sample_file = Path(__file__).parent.parent / "data" / "sample.csv"
    assert sample_file.exists()
    return pd.read_csv(sample_file)


def test_sample_dataset_integrity(sample_dataset):
    """Verify sample dataset dimensions and content."""
    assert len(sample_dataset) >= 50
    expected_cols = {
        "date", "product", "category", "region", "units_sold",
        "unit_price", "revenue", "advertising_spend", "customer_rating", "return_rate_pct"
    }
    assert expected_cols.issubset(set(sample_dataset.columns))


def test_end_to_end_profiling(sample_dataset):
    """Test full dataset profiling on sample dataset."""
    profile = profile_dataset(sample_dataset)

    # Check general
    assert profile["general"]["total_rows"] == len(sample_dataset)
    assert profile["general"]["total_columns"] == len(sample_dataset.columns)

    # Check numerical
    assert "revenue" in profile["numerical"]
    assert "advertising_spend" in profile["numerical"]
    assert profile["numerical"]["revenue"]["mean"] > 0

    # Check categorical
    assert "product" in profile["categorical"]
    assert "category" in profile["categorical"]
    assert "region" in profile["categorical"]

    # Check correlations
    assert profile["correlations"] is not None
    assert profile["correlations"].shape[0] >= 4

    # Check insights
    assert len(profile["insights"]) >= 3


@pytest.mark.parametrize("question,expected_op", [
    ("What is the average revenue?", "aggregate"),
    ("What is the total revenue?", "aggregate"),
    ("Which product generated the most revenue?", "groupby_aggregate"),
    ("What is the average customer_rating by category?", "groupby_aggregate"),
    ("What are the top 5 products by units_sold?", "groupby_aggregate"),
    ("Is advertising_spend related to revenue?", "correlation"),
    ("How did revenue change over time?", "time_series"),
    ("Show distribution of customer_rating", "distribution"),
    ("Which columns have missing values?", "missing_values"),
    ("Show outliers in revenue", "outliers"),
    ("How many rows and columns are in this dataset?", "dataset_info"),
    ("What are the column names and their data types?", "schema"),
])
def test_end_to_end_question_flow(sample_dataset, question, expected_op):
    """Test full NLQ -> Instruction -> Calculation -> Explanation -> Chart pipeline."""
    # 1. Parse Question
    instruction, is_llm, error = parse_question_with_llm(question, sample_dataset)
    assert instruction is not None
    assert instruction.get("operation") == expected_op

    # 2. Execute Calculation
    engine = AnalyticsEngine(sample_dataset)
    result = engine.execute(instruction)
    assert result.success is True
    assert result.error is None
    assert len(result.analysis_steps) > 0

    # 3. Generate Explanation
    explanation = explain_result_with_llm(question, instruction, result)
    assert explanation is not None
    assert len(explanation) > 10

    # 4. Generate Visual Chart
    if result.suggested_chart and result.suggested_chart.get("type") not in ("metric", "table"):
        chart = create_chart_from_result(result.suggested_chart, result.data, sample_dataset)
        assert chart is not None


def test_regression_dataset_dimensions(sample_dataset):
    """Regression test 1: Question asking for rows and columns returns exact dimensions."""
    question = "How many rows and columns are in this dataset?"
    inst, _, _ = parse_question_with_llm(question, sample_dataset)
    assert inst.get("operation") == "dataset_info"

    engine = AnalyticsEngine(sample_dataset)
    res = engine.execute(inst)
    assert res.success is True
    assert res.data["rows"] == len(sample_dataset)
    assert res.data["columns"] == len(sample_dataset.columns)

    explanation = explain_result_with_llm(question, inst, res)
    assert f"{len(sample_dataset):,} rows" in explanation or f"{len(sample_dataset)} rows" in explanation
    assert f"{len(sample_dataset.columns):,} columns" in explanation or f"{len(sample_dataset.columns)} columns" in explanation
    assert "revenue is" not in explanation


def test_regression_dataset_schema(sample_dataset):
    """Regression test 2: Question asking for column names and types returns complete schema."""
    question = "What are the column names and their data types?"
    inst, _, _ = parse_question_with_llm(question, sample_dataset)
    assert inst.get("operation") == "schema"

    engine = AnalyticsEngine(sample_dataset)
    res = engine.execute(inst)
    assert res.success is True
    assert isinstance(res.summary_table, pd.DataFrame)
    assert len(res.summary_table) == len(sample_dataset.columns)

    explanation = explain_result_with_llm(question, inst, res)
    for col in sample_dataset.columns:
        assert col in explanation
    assert "Missing values analysis completed" not in explanation


def test_regression_intent_classification_na_word_boundary(sample_dataset):
    """Regression test 3: Substring check does not confuse 'names' with missing values, but matches NA/null."""
    # Questions with words containing 'na' must NOT trigger missing_values
    inst_names, _, _ = parse_question_with_llm("What are the column names and their data types?", sample_dataset)
    assert inst_names.get("operation") == "schema"

    inst_national, _, _ = parse_question_with_llm("Show national distribution", sample_dataset)
    assert inst_national.get("operation") != "missing_values"

    # Standalone missing / null / NA questions MUST trigger missing_values
    inst_missing, _, _ = parse_question_with_llm("Are there any missing values?", sample_dataset)
    assert inst_missing.get("operation") == "missing_values"

    inst_nulls, _, _ = parse_question_with_llm("Are there any nulls in dataset?", sample_dataset)
    assert inst_nulls.get("operation") == "missing_values"

    inst_na, _, _ = parse_question_with_llm("Are there any NA values?", sample_dataset)
    assert inst_na.get("operation") == "missing_values"

    inst_slash_na, _, _ = parse_question_with_llm("Are there any N/A values?", sample_dataset)
    assert inst_slash_na.get("operation") == "missing_values"


def test_regression_top_n_explanation(sample_dataset):
    """Regression test 4: Top 5 products explanation contains all 5 ranked items and revenue figures."""
    question = "What are the top 5 products by total revenue?"
    inst, _, _ = parse_question_with_llm(question, sample_dataset)
    assert inst.get("operation") in ("groupby_aggregate", "top_n")
    assert inst.get("limit") == 5 or inst.get("n") == 5

    engine = AnalyticsEngine(sample_dataset)
    res = engine.execute(inst)
    assert res.success is True
    assert len(res.summary_table) == 5

    explanation = explain_result_with_llm(question, inst, res)

    # Must list all 5 products alongside their expected revenue values
    top_5_products = res.summary_table.iloc[:, 0].tolist()
    assert len(top_5_products) == 5
    for _, row in res.summary_table.iterrows():
        prod = row.iloc[0]
        val = row.iloc[1]
        val_str = f"{val:,.2f}"
        assert prod in explanation
        assert val_str in explanation
        assert f"**{prod}** ({val_str})" in explanation

    # Must NOT describe the 5 rows as the total number of product categories in dataset
    assert "grouped across 5 product categories" not in explanation


def test_regression_time_series_numerical_summary(sample_dataset):
    """Regression test 5: Time series explanation includes total revenue, peak period, and verified numbers."""
    question = "What is the total revenue, and how did revenue change month by month?"
    inst, _, _ = parse_question_with_llm(question, sample_dataset)
    assert inst.get("operation") == "time_series"

    engine = AnalyticsEngine(sample_dataset)
    res = engine.execute(inst)
    assert res.success is True
    assert len(res.summary_table) == 12  # 12 months in sample dataset

    explanation = explain_result_with_llm(question, inst, res)

    # Must include total revenue computed (4,710,473.00)
    total_rev = sample_dataset["revenue"].sum()
    assert f"{total_rev:,.2f}" in explanation
    assert "4,710,473.00" in explanation

    # Must include peak period (2025-11) and peak revenue (545,870.00)
    assert "Peak occurred in" in explanation
    assert "2025-11" in explanation
    assert "545,870.00" in explanation

    # Must include lowest period (2025-08) and lowest revenue (254,040.00)
    assert "lowest in" in explanation
    assert "2025-08" in explanation
    assert "254,040.00" in explanation

    assert "Check the chart and table below for details" not in explanation


def test_regression_count_formatting(sample_dataset):
    """Regression test 6: Count aggregations are displayed as integers without decimals."""
    question = "How many records are in this dataset?"
    inst = {"operation": "aggregate", "column": "revenue", "aggregation": "count"}
    engine = AnalyticsEngine(sample_dataset)
    res = engine.execute(inst)

    assert res.success is True
    assert res.data == len(sample_dataset)
    assert ".00" not in res.formatted_result

    explanation = explain_result_with_llm(question, inst, res)
    assert f"**{len(sample_dataset)}**" in explanation
    assert ".00" not in explanation


def test_streamlit_server_health():
    """Verify Streamlit server is active and returning 200 OK."""
    try:
        res = requests.get("http://localhost:8501/healthz", timeout=3)
        assert res.status_code == 200
    except requests.exceptions.ConnectionError:
        # If server is not running on 8501, try checking root
        res = requests.get("http://localhost:8501", timeout=3)
        assert res.status_code == 200
