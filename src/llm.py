"""LLM Translation and Explanation Module for AI Analytics Engineer.

Translates natural-language questions into strictly validated structured JSON analytical
instructions and explains computed Pandas results using verified facts.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.analyzer import AnalysisResult, resolve_column_name


SYSTEM_PROMPT_TRANSLATOR = """You are the Lead Analytical Brain of an AI Analytics Engine.
Your role is to translate a user's natural language question into a SINGLE structured JSON analytical instruction that will be executed by a deterministic Python/Pandas calculator.

CORE PRINCIPLES:
1. You do NOT write Python code. You output ONLY a JSON object matching our controlled analytical schema.
2. NEVER invent column names. Only use column names explicitly listed in the dataset metadata.
3. If a question is ambiguous, select the most reasonable analytical operation.
4. Output valid JSON only. Do not include markdown code blocks (```json ... ```) or conversational preamble.

SUPPORTED SCHEMAS:

1. Aggregate:
   {"operation": "aggregate", "column": "<col>", "aggregation": "mean"|"sum"|"median"|"min"|"max"|"count"|"std"|"nunique", "filter": [{"column": "<col>", "operator": "=="|"!="|">"|">="|"<"|"<="|"contains"|"in", "value": <val>}]}

2. GroupBy Aggregate / Ranking:
   {"operation": "groupby_aggregate", "group_column": "<categorical_col>", "value_column": "<numerical_col>", "aggregation": "sum"|"mean"|"count"|"median"|"min"|"max", "sort": "descending"|"ascending"|"none", "limit": <int or null>, "filter": <optional_filters>}

3. Top N / Bottom N:
   {"operation": "top_n", "group_column": "<categorical_col>", "column": "<numerical_col>", "n": <int>, "aggregation": "sum"|"mean"|"max"}
   {"operation": "bottom_n", "group_column": "<categorical_col>", "column": "<numerical_col>", "n": <int>, "aggregation": "sum"|"mean"|"min"}

4. Correlation:
   {"operation": "correlation", "column1": "<numerical_col>", "column2": "<optional_numerical_col>"}

5. Time Series Trend:
   {"operation": "time_series", "date_column": "<date_col>", "value_column": "<numerical_col>", "aggregation": "sum"|"mean", "time_granularity": "M"|"W"|"D"|"Q"|"Y"|"none", "group_by": "<optional_categorical_col>"}

6. Distribution:
   {"operation": "distribution", "column": "<col>"}

7. Missing Values:
   {"operation": "missing_values", "column": "<optional_col>"}

8. Outliers:
   {"operation": "outliers", "column": "<numerical_col>"}

9. Comparison:
   {"operation": "comparison", "group_column": "<categorical_col>", "value_column": "<numerical_col>", "groups": ["<val1>", "<val2>"], "aggregation": "mean"|"sum"}

10. Filter / Table Query:
    {"operation": "filter", "conditions": [{"column": "<col>", "operator": "=="|">"|"<"|etc, "value": <val>}], "select_columns": ["<col1>", "<col2>"], "limit": <int>}

11. Dataset Dimensions / Row and Column Counts:
    {"operation": "dataset_info"}

12. Column Names and Data Types / Schema:
    {"operation": "schema"}
"""

SYSTEM_PROMPT_EXPLAINER = """You are a senior data analyst presenting verified insights to business stakeholders.
You are given:
1. The user's original question
2. The exact analytical steps performed by Python
3. The verified computed numerical results from Python/Pandas

CRITICAL RULES:
- You must explain the result clearly, concisely, and professionally.
- NEVER invent, alter, or round numbers differently than the verified results. All numbers MUST come from the computed data.
- State the direct answer upfront, followed by any relevant context or business takeaway.
- Keep the response to 2-4 sentences unless the data contains a list or comparison.
"""


def extract_dataset_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """Extract token-efficient schema and summary statistics for LLM context."""
    columns_meta = []

    for col in df.columns:
        s = df[col]
        is_num = pd.api.types.is_numeric_dtype(s)
        is_date = pd.api.types.is_datetime64_any_dtype(s) or ("date" in str(col).lower() or "time" in str(col).lower())

        info = {
            "name": str(col),
            "type": "numerical" if is_num else "datetime" if is_date else "categorical",
            "non_null_count": int(s.count()),
        }

        if is_num:
            info["min"] = float(s.min()) if s.count() > 0 else None
            info["max"] = float(s.max()) if s.count() > 0 else None
            info["mean"] = round(float(s.mean()), 2) if s.count() > 0 else None
        else:
            sample_unique = s.dropna().unique()[:6].tolist()
            info["sample_values"] = [str(x) for x in sample_unique]
            info["unique_count"] = int(s.nunique())

        columns_meta.append(info)

    return {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "columns": columns_meta
    }


def rule_based_fallback_parser(question: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Smart heuristic parser for common questions when API key is unavailable."""
    q = question.lower().strip()
    columns = list(df.columns)
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    cat_cols = [c for c in columns if c not in num_cols]

    # 1. Dataset Dimensions / Shape / Row and Column Count
    if (
        re.search(r"\b(?:rows?\s+and\s+columns?|columns?\s+and\s+rows?|dataset\s+dimensions?|dataset\s+shape|dataset\s+size|how\s+many\s+rows|number\s+of\s+rows|row\s+count|total\s+rows|how\s+many\s+records|total\s+records)\b", q)
        or ("rows" in q and "columns" in q and ("how many" in q or "count" in q or "number" in q or "what" in q or "total" in q))
    ):
        return {"operation": "dataset_info"}

    # 2. Dataset Schema / Column Names & Types
    if (
        re.search(r"\b(?:data\s*types?|dtypes?|column\s+names?|columns?\s+and\s+types?|schema|list\s+(?:all\s+)?columns?)\b", q)
        or ("column" in q and ("types" in q or "dtypes" in q or "names" in q or "what are" in q or "list" in q))
    ):
        return {"operation": "schema"}

    # 3. Missing values (safely isolated with word boundaries)
    if re.search(r"\b(?:missing|null|nulls|nan|nans|na|n/a)\b", q):
        for col in columns:
            if re.search(r"\b" + re.escape(col.lower()) + r"\b", q):
                return {"operation": "missing_values", "column": col}
        return {"operation": "missing_values"}

    # 2. Outliers
    if "outlier" in q or "anomaly" in q or "anomalies" in q:
        for col in num_cols:
            if col.lower() in q:
                return {"operation": "outliers", "column": col}
        if num_cols:
            return {"operation": "outliers", "column": num_cols[0]}

    # 3. Correlation
    if "correlat" in q or "related" in q or "relationship" in q:
        matched_num = [c for c in num_cols if c.lower() in q]
        if len(matched_num) >= 2:
            return {"operation": "correlation", "column1": matched_num[0], "column2": matched_num[1]}
        elif len(matched_num) == 1:
            return {"operation": "correlation", "column1": matched_num[0]}
        elif len(num_cols) >= 2:
            return {"operation": "correlation", "column1": num_cols[0], "column2": num_cols[1]}

    # 4. Distribution / Histogram
    if "distribution" in q or "histogram" in q or "spread" in q or "frequency" in q:
        for col in columns:
            if col.lower() in q:
                return {"operation": "distribution", "column": col}
        if num_cols:
            return {"operation": "distribution", "column": num_cols[0]}

    # 5. Time series / Over time / Trend
    if "over time" in q or "trend" in q or "by month" in q or "monthly" in q or "by date" in q or "daily" in q:
        date_col = next((c for c in columns if "date" in c.lower() or "time" in c.lower()), None)
        val_col = next((c for c in num_cols if c.lower() in q), num_cols[0] if num_cols else None)
        if date_col and val_col:
            granularity = "M" if "month" in q else "D"
            return {
                "operation": "time_series",
                "date_column": date_col,
                "value_column": val_col,
                "aggregation": "sum",
                "time_granularity": granularity
            }

    # 6. Top N / Bottom N / Highest / Lowest Grouping
    # e.g. "Which product generated the most revenue?", "top 5 categories by sales"
    top_n_match = re.search(r"top\s+(\d+)", q)
    bottom_n_match = re.search(r"(?:bottom|lowest)\s+(\d+)", q)
    n_val = int(top_n_match.group(1)) if top_n_match else (int(bottom_n_match.group(1)) if bottom_n_match else 1)

    is_highest = any(w in q for w in ["most", "highest", "top", "best", "max", "maximum", "greatest"])
    is_lowest = any(w in q for w in ["least", "lowest", "bottom", "worst", "min", "minimum"])

    # Identify target value column and group column
    target_val_col = next((c for c in num_cols if c.lower() in q), None)
    target_group_col = next((c for c in cat_cols if c.lower() in q), None)

    # Fallback to defaults if "revenue", "sales", "price" matches
    if not target_val_col and num_cols:
        for w in ["revenue", "sales", "price", "amount", "cost", "units", "rating", "spend"]:
            found = next((c for c in num_cols if w in c.lower()), None)
            if found:
                target_val_col = found
                break
        if not target_val_col:
            target_val_col = num_cols[0]

    if (is_highest or is_lowest or top_n_match or bottom_n_match or "by " in q or "per " in q or "which " in q):
        if not target_group_col and cat_cols:
            # Check for words like product, category, region
            for w in ["product", "category", "region", "department", "item", "customer", "country"]:
                found = next((c for c in cat_cols if w in c.lower()), None)
                if found:
                    target_group_col = found
                    break
            if not target_group_col:
                target_group_col = cat_cols[0]

        if target_group_col and target_val_col:
            agg = "mean" if any(w in q for w in ["average", "avg", "mean", "rating"]) else "sum"
            sort_dir = "ascending" if is_lowest else "descending"
            limit_val = n_val if (top_n_match or bottom_n_match or n_val > 1) else (1 if ("which" in q or is_highest or is_lowest) else None)
            return {
                "operation": "groupby_aggregate",
                "group_column": target_group_col,
                "value_column": target_val_col,
                "aggregation": agg,
                "sort": sort_dir,
                "limit": limit_val
            }

    # 7. Single Column Aggregations (mean, sum, median, min, max, count)
    agg_keyword_map = {
        "average": "mean", "avg": "mean", "mean": "mean",
        "total": "sum", "sum": "sum",
        "median": "median",
        "minimum": "min", "min": "min",
        "maximum": "max", "max": "max",
        "how many": "count", "count": "count",
        "unique": "nunique", "distinct": "nunique"
    }

    detected_agg = None
    for k, v in agg_keyword_map.items():
        if k in q:
            detected_agg = v
            break

    if detected_agg and target_val_col:
        return {
            "operation": "aggregate",
            "column": target_val_col,
            "aggregation": detected_agg
        }
    elif target_val_col:
        return {
            "operation": "aggregate",
            "column": target_val_col,
            "aggregation": "mean"
        }

    # Default fallback: general summary of first column
    first_col = columns[0] if columns else "unknown"
    return {
        "operation": "aggregate",
        "column": first_col,
        "aggregation": "count"
    }


def parse_question_with_llm(
    question: str,
    df: pd.DataFrame,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini"
) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    """Translate natural-language query into structured JSON instruction.

    Returns:
        tuple (instruction_dict, is_llm_generated, error_message)
    """
    metadata = extract_dataset_metadata(df)

    # Check if OpenAI API key is available
    active_key = api_key or os.getenv("OPENAI_API_KEY")
    if not active_key or active_key.strip() == "" or active_key == "your_openai_api_key_here":
        # Gracefully use rule-based parser
        instruction = rule_based_fallback_parser(question, df)
        return instruction, False, None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=active_key)

        user_prompt = f"""Dataset Metadata:
{json.dumps(metadata, indent=2)}

User Question:
"{question}"

Generate the structured JSON analytical instruction:"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_TRANSLATOR},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        content = response.choices[0].message.content
        instruction = json.loads(content)
        return instruction, True, None

    except Exception as e:
        # Fallback to heuristic parser if API call fails
        fallback_inst = rule_based_fallback_parser(question, df)
        return fallback_inst, False, f"LLM translation fallback (API error: {str(e)})"


def explain_result_with_llm(
    question: str,
    instruction: Dict[str, Any],
    result: AnalysisResult,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini"
) -> str:
    """Generate professional explanation grounded in computed Pandas result."""
    if not result.success:
        return f"Could not complete analysis: {result.error}"

    active_key = api_key or os.getenv("OPENAI_API_KEY")

    # Prepare result summary
    if isinstance(result.summary_table, pd.DataFrame) and not result.summary_table.empty:
        summary_repr = result.summary_table.head(10).to_string(index=False)
    else:
        summary_repr = result.formatted_result

    if not active_key or active_key.strip() == "" or active_key == "your_openai_api_key_here":
        # High quality deterministic explanation
        return generate_deterministic_explanation(question, instruction, result)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=active_key)

        user_prompt = f"""User Question:
"{question}"

Structured Operation Performed:
{json.dumps(instruction, indent=2)}

Calculated Result from Python:
{summary_repr}

Steps Performed:
{json.dumps(result.analysis_steps, indent=2)}

Provide a concise, direct, professional explanation using ONLY the verified numbers above."""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_EXPLAINER},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=250
        )
        return response.choices[0].message.content.strip()

    except Exception:
        return generate_deterministic_explanation(question, instruction, result)


def generate_deterministic_explanation(
    question: str,
    instruction: Dict[str, Any],
    result: AnalysisResult
) -> str:
    """Fallback generator that constructs exact factual explanations without LLM."""
    if not result.success:
        return f"Could not complete analysis: {result.error}"

    op = instruction.get("operation", "")

    if op == "aggregate":
        col = instruction.get("column")
        agg = instruction.get("aggregation", "mean")
        val = result.data
        if agg in ("count", "nunique") or isinstance(val, (int, np.integer)):
            formatted_val = f"{int(val):,}"
        elif isinstance(val, (float, np.floating)):
            formatted_val = f"{val:,.2f}"
        else:
            formatted_val = str(val)
        return f"The computed {agg} for **{col}** is **{formatted_val}**."

    elif op in ("groupby_aggregate", "ranking", "top_n", "bottom_n"):
        group_col = instruction.get("group_column") or instruction.get("item_column")
        val_col = instruction.get("value_column") or instruction.get("column")
        agg = instruction.get("aggregation", "sum")
        df_tbl = result.summary_table if isinstance(result.summary_table, pd.DataFrame) else result.data
        if isinstance(df_tbl, pd.DataFrame) and not df_tbl.empty:
            limit = instruction.get("limit") or instruction.get("n") or instruction.get("top_n")

            if limit and int(limit) > 1 and len(df_tbl) > 1:
                ranked_items = []
                for idx, (_, r) in enumerate(df_tbl.iterrows(), 1):
                    cat_name = r.iloc[0]
                    val_raw = r.iloc[1]
                    if agg in ("count", "nunique") or isinstance(val_raw, (int, np.integer)):
                        val_str = f"{int(val_raw):,}"
                    elif isinstance(val_raw, (float, np.floating)):
                        val_str = f"{val_raw:,.2f}"
                    else:
                        val_str = str(val_raw)
                    ranked_items.append(f"{idx}. **{cat_name}** ({val_str})")
                items_str = ", ".join(ranked_items)
                return f"Top {len(df_tbl)} **{group_col}** by {agg} **{val_col}**: {items_str}."

            top_row = df_tbl.iloc[0]
            cat = top_row.iloc[0]
            val = top_row.iloc[1]
            formatted_val = f"{int(val):,}" if (agg in ("count", "nunique") or isinstance(val, (int, np.integer))) else (f"{val:,.2f}" if isinstance(val, (float, np.floating)) else str(val))

            if limit == 1 or "which" in question.lower() or "highest" in question.lower() or "most" in question.lower() or "lowest" in question.lower() or "least" in question.lower():
                superlative = "lowest" if ("lowest" in question.lower() or "least" in question.lower() or str(instruction.get("sort")).lower() in ("asc", "ascending")) else "highest"
                return f"**{cat}** has the {superlative} {agg} {val_col} at **{formatted_val}**."
            else:
                return f"Aggregated {agg} {val_col} across {len(df_tbl)} {group_col} categories. Top performer is **{cat}** with **{formatted_val}**."
        return result.formatted_result

    elif op == "correlation":
        return result.formatted_result

    elif op == "time_series":
        date_col = instruction.get("date_column")
        val_col = instruction.get("value_column")
        agg = instruction.get("aggregation", "sum")
        df_tbl = result.summary_table if isinstance(result.summary_table, pd.DataFrame) else result.data

        if isinstance(df_tbl, pd.DataFrame) and not df_tbl.empty:
            period_col = df_tbl.columns[0]
            metric_col = df_tbl.columns[1] if len(df_tbl.columns) > 1 else period_col

            total_val = float(df_tbl[metric_col].sum())
            total_str = f"{total_val:,.2f}"

            max_idx = df_tbl[metric_col].idxmax()
            peak_period = str(df_tbl.loc[max_idx, period_col])
            peak_val = float(df_tbl.loc[max_idx, metric_col])
            peak_str = f"{peak_val:,.2f}"

            min_idx = df_tbl[metric_col].idxmin()
            min_period = str(df_tbl.loc[min_idx, period_col])
            min_val = float(df_tbl.loc[min_idx, metric_col])
            min_str = f"{min_val:,.2f}"

            num_periods = len(df_tbl)

            if agg == "sum":
                return (
                    f"Total **{val_col}** across {num_periods} periods is **{total_str}**. "
                    f"Peak occurred in **{peak_period}** (**{peak_str}**), and lowest in **{min_period}** (**{min_str}**)."
                )
            else:
                avg_val = float(df_tbl[metric_col].mean())
                return (
                    f"Average **{val_col}** across {num_periods} periods is **{avg_val:,.2f}** (Total: **{total_str}**). "
                    f"Peak occurred in **{peak_period}** (**{peak_str}**), and lowest in **{min_period}** (**{min_str}**)."
                )
        return result.formatted_result

    elif op == "distribution":
        col = instruction.get("column")
        return f"Analyzed distribution for **{col}**. {result.formatted_result}"

    elif op == "missing_values":
        return f"Missing values analysis completed: {result.formatted_result}"

    elif op == "outliers":
        col = instruction.get("column")
        return f"Outlier scan for **{col}**: {result.formatted_result}"

    elif op in ("dataset_info", "shape", "dimensions"):
        if isinstance(result.data, dict):
            rows = result.data.get("rows", len(result.summary_table) if isinstance(result.summary_table, pd.DataFrame) else 0)
            cols = result.data.get("columns", len(result.summary_table.columns) if isinstance(result.summary_table, pd.DataFrame) else 0)
            return f"The dataset contains **{rows:,} rows** and **{cols:,} columns**."
        return result.formatted_result

    elif op in ("schema", "columns_info", "data_types"):
        if isinstance(result.summary_table, pd.DataFrame) and not result.summary_table.empty:
            cols_preview = []
            for _, r in result.summary_table.iterrows():
                col_name = r.get("Column Name", r.iloc[0])
                dtype = r.get("Data Type", r.iloc[1])
                cols_preview.append(f"**{col_name}** (`{dtype}`)")
            return f"The dataset has {len(result.summary_table)} columns: " + ", ".join(cols_preview) + "."
        return result.formatted_result

    return result.formatted_result
