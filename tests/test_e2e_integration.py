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
    if result.suggested_chart and result.suggested_chart.get("type") != "metric":
        chart = create_chart_from_result(result.suggested_chart, result.data, sample_dataset)
        assert chart is not None


def test_streamlit_server_health():
    """Verify Streamlit server is active and returning 200 OK."""
    try:
        res = requests.get("http://localhost:8501/healthz", timeout=3)
        assert res.status_code == 200
    except requests.exceptions.ConnectionError:
        # If server is not running on 8501, try checking root
        res = requests.get("http://localhost:8501", timeout=3)
        assert res.status_code == 200
