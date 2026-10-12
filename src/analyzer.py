"""Deterministic Analytical Engine for AI Analytics Engineer.

Executes structured analytical instructions using Pandas.
Ensures zero arbitrary code execution and guarantees mathematical precision.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


@dataclass
class AnalysisResult:
    """Standard container for analysis results."""
    success: bool
    operation: str
    data: Any = None
    summary_table: Optional[pd.DataFrame] = None
    formatted_result: str = ""
    analysis_steps: List[str] = field(default_factory=list)
    suggested_chart: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dictionary."""
        return {
            "success": self.success,
            "operation": self.operation,
            "formatted_result": self.formatted_result,
            "analysis_steps": self.analysis_steps,
            "suggested_chart": self.suggested_chart,
            "error": self.error,
        }


def resolve_column_name(col_name: Optional[str], df: pd.DataFrame) -> Optional[str]:
    """Resolve column name with case-insensitivity and whitespace stripping."""
    if not col_name or not isinstance(col_name, str):
        return None

    cleaned = col_name.strip()
    if cleaned in df.columns:
        return cleaned

    # Case-insensitive match
    lower_map = {c.lower(): c for c in df.columns}
    if cleaned.lower() in lower_map:
        return lower_map[cleaned.lower()]

    # Match with underscores / spaces replaced
    normalized = cleaned.lower().replace(" ", "_").replace("-", "_")
    norm_map = {c.lower().replace(" ", "_").replace("-", "_"): c for c in df.columns}
    if normalized in norm_map:
        return norm_map[normalized]

    return None


def apply_filters(df: pd.DataFrame, filters: Union[Dict[str, Any], List[Dict[str, Any]]]) -> tuple[pd.DataFrame, List[str]]:
    """Apply structured filter conditions safely to DataFrame."""
    filtered_df = df.copy()
    filter_steps = []

    if not filters:
        return filtered_df, filter_steps

    if isinstance(filters, dict):
        filter_list = [filters]
    elif isinstance(filters, list):
        filter_list = filters
    else:
        raise ValueError(
            f"Invalid filter format: expected a dictionary or list of dictionaries, got {type(filters).__name__}."
        )

    for f in filter_list:
        if not isinstance(f, dict):
            raise ValueError(
                f"Invalid filter condition: expected a dictionary, got {type(f).__name__}."
            )

        raw_col = f.get("column")
        col = resolve_column_name(raw_col, df)
        if not col:
            raise ValueError(f"Filter column '{raw_col}' not found in dataset.")

        op = str(f.get("operator", "==")).strip().lower()
        val = f.get("value")

        if val is None and op not in ("isna", "is_null", "notna", "not_null"):
            continue

        series = filtered_df[col]

        if op in ("==", "=", "eq"):
            if pd.api.types.is_numeric_dtype(series):
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    pass
            mask = series == val
            filter_steps.append(f"Filter where `{col}` == '{val}'")
            filtered_df = filtered_df[mask]
        elif op in ("!=", "<>", "ne"):
            mask = series != val
            filter_steps.append(f"Filter where `{col}` != '{val}'")
            filtered_df = filtered_df[mask]
        elif op in (">", "gt"):
            mask = series > float(val)
            filter_steps.append(f"Filter where `{col}` > {val}")
            filtered_df = filtered_df[mask]
        elif op in (">=", "gte"):
            mask = series >= float(val)
            filter_steps.append(f"Filter where `{col}` >= {val}")
            filtered_df = filtered_df[mask]
        elif op in ("<", "lt"):
            mask = series < float(val)
            filter_steps.append(f"Filter where `{col}` < {val}")
            filtered_df = filtered_df[mask]
        elif op in ("<=", "lte"):
            mask = series <= float(val)
            filter_steps.append(f"Filter where `{col}` <= {val}")
            filtered_df = filtered_df[mask]
        elif op in ("contains", "like"):
            mask = series.astype(str).str.contains(str(val), case=False, na=False, regex=False)
            filter_steps.append(f"Filter where `{col}` contains '{val}'")
            filtered_df = filtered_df[mask]
        elif op in ("in", "isin"):
            if isinstance(val, (list, tuple, set)):
                val_list = list(val)
            else:
                val_list = [val]
            mask = series.isin(val_list)
            filter_steps.append(f"Filter where `{col}` in {val_list}")
            filtered_df = filtered_df[mask]
        elif op in ("isna", "is_null"):
            mask = series.isna()
            filter_steps.append(f"Filter where `{col}` is null")
            filtered_df = filtered_df[mask]
        elif op in ("notna", "not_null"):
            mask = series.notna()
            filter_steps.append(f"Filter where `{col}` is not null")
            filtered_df = filtered_df[mask]
        else:
            raise ValueError(f"Unsupported filter operator '{op}'.")

    return filtered_df, filter_steps


class AnalyticsEngine:
    """Deterministic analytics engine executing structured analytical instructions."""

    def __init__(self, df: pd.DataFrame):
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("AnalyticsEngine requires a valid pandas DataFrame.")
        self.df = df.copy()

    def execute(self, instruction: Dict[str, Any]) -> AnalysisResult:
        """Validate and dispatch instruction to appropriate handler."""
        if not isinstance(instruction, dict):
            return AnalysisResult(
                success=False,
                operation="unknown",
                error="Invalid instruction format. Expected a dictionary schema."
            )

        op = str(instruction.get("operation", "")).strip().lower()

        handler_map = {
            "aggregate": self._op_aggregate,
            "groupby_aggregate": self._op_groupby_aggregate,
            "top_n": self._op_top_n,
            "bottom_n": self._op_bottom_n,
            "filter": self._op_filter,
            "correlation": self._op_correlation,
            "time_series": self._op_time_series,
            "distribution": self._op_distribution,
            "missing_values": self._op_missing_values,
            "outliers": self._op_outliers,
            "comparison": self._op_comparison,
            "ranking": self._op_ranking,
            "dataset_info": self._op_dataset_info,
            "shape": self._op_dataset_info,
            "dimensions": self._op_dataset_info,
            "schema": self._op_schema,
            "columns_info": self._op_schema,
            "data_types": self._op_schema,
        }

        handler = handler_map.get(op)
        if not handler:
            supported = ", ".join(handler_map.keys())
            return AnalysisResult(
                success=False,
                operation=op,
                error=f"Unsupported operation '{op}'. Supported operations are: {supported}"
            )

        try:
            return handler(instruction)
        except Exception as e:
            return AnalysisResult(
                success=False,
                operation=op,
                error=f"Calculation error during '{op}': {str(e)}"
            )

    def _op_aggregate(self, inst: Dict[str, Any]) -> AnalysisResult:
        """Compute single-column aggregation (mean, sum, median, etc.)."""
        raw_col = inst.get("column")
        col = resolve_column_name(raw_col, self.df)
        if not col:
            return AnalysisResult(
                success=False,
                operation="aggregate",
                error=f"Column '{raw_col}' not found in dataset."
            )

        agg_func = str(inst.get("aggregation", "mean")).strip().lower()
        allowed_aggs = {
            "mean": "mean", "avg": "mean", "average": "mean",
            "sum": "sum", "total": "sum",
            "median": "median", "med": "median",
            "min": "min", "minimum": "min",
            "max": "max", "maximum": "max",
            "count": "count",
            "std": "std", "standard_deviation": "std",
            "var": "var", "variance": "var",
            "nunique": "nunique", "unique_count": "nunique"
        }

        if agg_func not in allowed_aggs:
            return AnalysisResult(
                success=False,
                operation="aggregate",
                error=f"Unsupported aggregation '{agg_func}'. Allowed: mean, sum, median, min, max, count, std, nunique."
            )

        mapped_agg = allowed_aggs[agg_func]
        df_filtered, filter_steps = apply_filters(self.df, inst.get("filter"))

        series = df_filtered[col]
        steps = list(filter_steps)
        steps.append(f"Calculate `{mapped_agg.upper()}` on column `{col}`")

        if mapped_agg in ("mean", "sum", "median", "min", "max", "std", "var"):
            if not pd.api.types.is_numeric_dtype(series):
                return AnalysisResult(
                    success=False,
                    operation="aggregate",
                    error=f"Column '{col}' is not numerical. Cannot compute '{mapped_agg}'."
                )
            res_val = getattr(series.dropna(), mapped_agg)()
        elif mapped_agg == "count":
            res_val = int(series.dropna().count())
        elif mapped_agg == "nunique":
            res_val = int(series.nunique())

        res_val_formatted = round(float(res_val), 4) if isinstance(res_val, (float, np.floating)) else res_val

        if mapped_agg in ("count", "nunique") or isinstance(res_val_formatted, (int, np.integer)):
            formatted_res = f"{mapped_agg.upper()}({col}) = {int(res_val_formatted):,}"
        elif isinstance(res_val_formatted, (float, np.floating)):
            formatted_res = f"{mapped_agg.upper()}({col}) = {res_val_formatted:,.2f}"
        else:
            formatted_res = str(res_val_formatted)

        summary_df = pd.DataFrame([{
            "Column": col,
            "Aggregation": mapped_agg.upper(),
            "Result": res_val_formatted
        }])

        return AnalysisResult(
            success=True,
            operation="aggregate",
            data=res_val_formatted,
            summary_table=summary_df,
            formatted_result=formatted_res,
            analysis_steps=steps,
            suggested_chart={
                "type": "metric",
                "label": f"{mapped_agg.capitalize()} of {col}",
                "value": res_val_formatted
            }
        )

    def _op_groupby_aggregate(self, inst: Dict[str, Any]) -> AnalysisResult:
        """Group by categorical column(s) and aggregate numerical column."""
        raw_group = inst.get("group_column") or inst.get("group_columns") or inst.get("by")
        raw_val = inst.get("value_column") or inst.get("column")

        group_col = resolve_column_name(raw_group, self.df)
        val_col = resolve_column_name(raw_val, self.df)

        if not group_col:
            return AnalysisResult(
                success=False,
                operation="groupby_aggregate",
                error=f"Grouping column '{raw_group}' not found in dataset."
            )

        agg_func = str(inst.get("aggregation", "sum")).strip().lower()
        allowed_aggs = {
            "sum": "sum", "total": "sum",
            "mean": "mean", "avg": "mean", "average": "mean",
            "count": "count",
            "median": "median",
            "min": "min", "max": "max",
            "std": "std", "nunique": "nunique"
        }

        if agg_func not in allowed_aggs:
            return AnalysisResult(
                success=False,
                operation="groupby_aggregate",
                error=f"Unsupported aggregation '{agg_func}'."
            )
        mapped_agg = allowed_aggs[agg_func]

        if not val_col:
            if mapped_agg == "count":
                val_col = group_col
            else:
                return AnalysisResult(
                    success=False,
                    operation="groupby_aggregate",
                    error=f"Value column '{raw_val}' not found in dataset."
                )

        df_filtered, filter_steps = apply_filters(self.df, inst.get("filter"))
        steps = list(filter_steps)
        steps.append(f"Group dataset by `{group_col}` and calculate `{mapped_agg.upper()}({val_col})`")

        if mapped_agg == "count":
            grouped = df_filtered.groupby(group_col)[val_col].count()
        else:
            if not pd.api.types.is_numeric_dtype(df_filtered[val_col]):
                return AnalysisResult(
                    success=False,
                    operation="groupby_aggregate",
                    error=f"Value column '{val_col}' must be numerical for '{mapped_agg}' aggregation."
                )
            grouped = getattr(df_filtered.groupby(group_col)[val_col], mapped_agg)()

        result_col_name = f"{mapped_agg.lower()}_{val_col}" if val_col != group_col else f"{mapped_agg.lower()}_count"
        res_df = grouped.reset_index().rename(columns={val_col: result_col_name})

        # Sorting
        sort_order = str(inst.get("sort", "descending")).strip().lower()
        if sort_order in ("desc", "descending"):
            res_df = res_df.sort_values(by=result_col_name, ascending=False)
            steps.append(f"Sort descending by `{result_col_name}`")
        elif sort_order in ("asc", "ascending"):
            res_df = res_df.sort_values(by=result_col_name, ascending=True)
            steps.append(f"Sort ascending by `{result_col_name}`")

        # Limit / Top N
        limit = inst.get("limit") or inst.get("top_n")
        if limit:
            try:
                limit_int = int(limit)
                if limit_int > 0:
                    res_df = res_df.head(limit_int)
                    steps.append(f"Limit to top {limit_int} rows")
            except (ValueError, TypeError):
                pass

        # Format results
        if not res_df.empty:
            limit_val = inst.get("limit") or inst.get("top_n") or inst.get("n")
            if limit_val and int(limit_val) > 1 and len(res_df) > 1:
                ranked_items = []
                for idx, (_, r) in enumerate(res_df.iterrows(), 1):
                    val = r[result_col_name]
                    formatted_val = f"{int(val):,}" if (mapped_agg in ("count", "nunique") or isinstance(val, (int, np.integer))) else (f"{val:,.2f}" if isinstance(val, (float, np.floating)) else str(val))
                    ranked_items.append(f"{idx}. {r[group_col]} ({formatted_val})")
                formatted_res = f"Top {len(res_df)} {group_col} by {mapped_agg.upper()}({val_col}): " + ", ".join(ranked_items)
            else:
                top_row = res_df.iloc[0]
                top_cat = top_row[group_col]
                top_val = top_row[result_col_name]
                formatted_val = f"{int(top_val):,}" if (mapped_agg in ("count", "nunique") or isinstance(top_val, (int, np.integer))) else (f"{top_val:,.2f}" if isinstance(top_val, (float, np.floating)) else str(top_val))
                formatted_res = f"Top result: {group_col} '{top_cat}' with {mapped_agg.upper()}({val_col}) = {formatted_val}"
        else:
            formatted_res = "No matching records found."

        return AnalysisResult(
            success=True,
            operation="groupby_aggregate",
            data=res_df,
            summary_table=res_df,
            formatted_result=formatted_res,
            analysis_steps=steps,
            suggested_chart={
                "type": "bar",
                "x": group_col,
                "y": result_col_name,
                "title": f"{mapped_agg.capitalize()} of {val_col} by {group_col}",
            }
        )

    def _op_top_n(self, inst: Dict[str, Any]) -> AnalysisResult:
        """Find Top N items (either aggregated by group or raw rows)."""
        inst_copy = inst.copy()
        inst_copy["sort"] = "descending"
        if "n" in inst_copy and "limit" not in inst_copy:
            inst_copy["limit"] = inst_copy["n"]

        raw_group = inst.get("group_column") or inst.get("item_column")
        if raw_group:
            inst_copy["operation"] = "groupby_aggregate"
            if not inst_copy.get("aggregation"):
                inst_copy["aggregation"] = "sum"
            return self._op_groupby_aggregate(inst_copy)

        # Top N raw rows
        raw_val = inst.get("column") or inst.get("value_column")
        val_col = resolve_column_name(raw_val, self.df)
        if not val_col:
            return AnalysisResult(
                success=False,
                operation="top_n",
                error=f"Column '{raw_val}' not found."
            )

        n = int(inst.get("n", inst.get("limit", 5)))
        df_filtered, filter_steps = apply_filters(self.df, inst.get("filter"))
        steps = list(filter_steps)
        steps.append(f"Sort rows descending by `{val_col}` and select top {n}")

        top_df = df_filtered.sort_values(by=val_col, ascending=False).head(n)

        return AnalysisResult(
            success=True,
            operation="top_n",
            data=top_df,
            summary_table=top_df,
            formatted_result=f"Top {n} records by {val_col}",
            analysis_steps=steps,
            suggested_chart={
                "type": "table",
                "title": f"Top {n} Rows by {val_col}"
            }
        )

    def _op_bottom_n(self, inst: Dict[str, Any]) -> AnalysisResult:
        """Find Bottom N items."""
        inst_copy = inst.copy()
        inst_copy["sort"] = "ascending"
        if "n" in inst_copy and "limit" not in inst_copy:
            inst_copy["limit"] = inst_copy["n"]

        raw_group = inst.get("group_column") or inst.get("item_column")
        if raw_group:
            inst_copy["operation"] = "groupby_aggregate"
            if not inst_copy.get("aggregation"):
                inst_copy["aggregation"] = "sum"
            return self._op_groupby_aggregate(inst_copy)

        raw_val = inst.get("column") or inst.get("value_column")
        val_col = resolve_column_name(raw_val, self.df)
        if not val_col:
            return AnalysisResult(
                success=False,
                operation="bottom_n",
                error=f"Column '{raw_val}' not found."
            )

        n = int(inst.get("n", inst.get("limit", 5)))
        df_filtered, filter_steps = apply_filters(self.df, inst.get("filter"))
        steps = list(filter_steps)
        steps.append(f"Sort rows ascending by `{val_col}` and select bottom {n}")

        bottom_df = df_filtered.sort_values(by=val_col, ascending=True).head(n)

        return AnalysisResult(
            success=True,
            operation="bottom_n",
            data=bottom_df,
            summary_table=bottom_df,
            formatted_result=f"Bottom {n} records by {val_col}",
            analysis_steps=steps,
            suggested_chart={
                "type": "table",
                "title": f"Bottom {n} Rows by {val_col}"
            }
        )

    def _op_filter(self, inst: Dict[str, Any]) -> AnalysisResult:
        """Filter dataset and optionally select columns."""
        conditions = inst.get("conditions", inst.get("filter"))
        df_filtered, filter_steps = apply_filters(self.df, conditions)
        steps = list(filter_steps)

        select_cols = inst.get("select_columns")
        if select_cols and isinstance(select_cols, list):
            resolved_cols = [resolve_column_name(c, df_filtered) for c in select_cols]
            valid_cols = [c for c in resolved_cols if c]
            if valid_cols:
                df_filtered = df_filtered[valid_cols]
                steps.append(f"Selected columns: {', '.join(valid_cols)}")

        limit = inst.get("limit")
        if limit:
            try:
                df_filtered = df_filtered.head(int(limit))
                steps.append(f"Limit output to {limit} rows")
            except (ValueError, TypeError):
                pass

        return AnalysisResult(
            success=True,
            operation="filter",
            data=df_filtered,
            summary_table=df_filtered,
            formatted_result=f"Filtered result returned {len(df_filtered):,} matching rows.",
            analysis_steps=steps,
            suggested_chart={"type": "table", "title": "Filtered Results"}
        )

    def _op_correlation(self, inst: Dict[str, Any]) -> AnalysisResult:
        """Calculate Pearson correlation between two numerical columns or matrix."""
        raw_col1 = inst.get("column1") or inst.get("column")
        raw_col2 = inst.get("column2")

        col1 = resolve_column_name(raw_col1, self.df)
        col2 = resolve_column_name(raw_col2, self.df)

        if not col1:
            return AnalysisResult(
                success=False,
                operation="correlation",
                error=f"Column '{raw_col1}' not found in dataset."
            )

        if not pd.api.types.is_numeric_dtype(self.df[col1]):
            return AnalysisResult(
                success=False,
                operation="correlation",
                error=f"Column '{col1}' is not numerical."
            )

        if col2:
            if not pd.api.types.is_numeric_dtype(self.df[col2]):
                return AnalysisResult(
                    success=False,
                    operation="correlation",
                    error=f"Column '{col2}' is not numerical."
                )

            clean_df = self.df[[col1, col2]].dropna()
            if len(clean_df) < 2:
                return AnalysisResult(
                    success=False,
                    operation="correlation",
                    error="Not enough valid data points to compute correlation."
                )

            corr_val = float(clean_df[col1].corr(clean_df[col2]))
            steps = [f"Compute Pearson correlation coefficient between `{col1}` and `{col2}`"]

            direction = "positive" if corr_val > 0 else "negative"
            strength = "strong" if abs(corr_val) >= 0.7 else "moderate" if abs(corr_val) >= 0.4 else "weak"

            res_df = pd.DataFrame([{
                "Variable 1": col1,
                "Variable 2": col2,
                "Correlation (r)": round(corr_val, 4),
                "Relationship": f"{strength} {direction}"
            }])

            return AnalysisResult(
                success=True,
                operation="correlation",
                data=corr_val,
                summary_table=res_df,
                formatted_result=f"Pearson correlation between '{col1}' and '{col2}' is r = {corr_val:+.3f} ({strength} {direction}).",
                analysis_steps=steps,
                suggested_chart={
                    "type": "scatter",
                    "x": col1,
                    "y": col2,
                    "title": f"Correlation: {col1} vs {col2} (r = {corr_val:+.2f})"
                }
            )
        else:
            # Correlation of col1 vs all numerical columns
            num_cols = [c for c in self.df.select_dtypes(include=[np.number]).columns if c != col1]
            if not num_cols:
                return AnalysisResult(
                    success=False,
                    operation="correlation",
                    error="No other numerical columns found to compute correlation against."
                )

            corrs = []
            for other_col in num_cols:
                clean_df = self.df[[col1, other_col]].dropna()
                if len(clean_df) >= 2:
                    r = float(clean_df[col1].corr(clean_df[other_col]))
                    corrs.append({
                        "Column": other_col,
                        "Correlation_r": round(r, 4),
                        "Abs_r": round(abs(r), 4)
                    })

            corr_df = pd.DataFrame(corrs).sort_values(by="Abs_r", ascending=False).drop(columns=["Abs_r"])
            steps = [f"Compute correlation of `{col1}` with all other numerical columns"]

            return AnalysisResult(
                success=True,
                operation="correlation",
                data=corr_df,
                summary_table=corr_df,
                formatted_result=f"Correlations for {col1} computed across {len(corr_df)} variables.",
                analysis_steps=steps,
                suggested_chart={
                    "type": "bar",
                    "x": "Column",
                    "y": "Correlation_r",
                    "title": f"Correlations with {col1}"
                }
            )

    def _op_time_series(self, inst: Dict[str, Any]) -> AnalysisResult:
        """Analyze metric over time with automatic date parsing and frequency aggregation."""
        raw_date = inst.get("date_column")
        raw_val = inst.get("value_column")

        date_col = resolve_column_name(raw_date, self.df)
        val_col = resolve_column_name(raw_val, self.df)

        # If date column not explicitly supplied, find first datetime or date-like column
        if not date_col:
            for c in self.df.columns:
                if "date" in c.lower() or "time" in c.lower() or "month" in c.lower() or "year" in c.lower():
                    date_col = c
                    break

        if not date_col:
            return AnalysisResult(
                success=False,
                operation="time_series",
                error="No valid date/time column specified or detected in the dataset."
            )

        if not val_col:
            return AnalysisResult(
                success=False,
                operation="time_series",
                error=f"Value column '{raw_val}' not found in dataset."
            )

        agg_func = str(inst.get("aggregation", "sum")).strip().lower()
        if agg_func not in ("sum", "mean", "count", "median", "min", "max"):
            agg_func = "sum"

        df_filtered, filter_steps = apply_filters(self.df, inst.get("filter"))
        steps = list(filter_steps)
        steps.append(f"Convert `{date_col}` to Datetime format")

        # Safe datetime conversion
        ts_df = df_filtered.copy()
        ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
        ts_df = ts_df.dropna(subset=[date_col, val_col])

        if ts_df.empty:
            return AnalysisResult(
                success=False,
                operation="time_series",
                error=f"Could not parse valid dates in column '{date_col}'."
            )

        granularity = str(inst.get("time_granularity", "none")).upper()
        if granularity in ("M", "MONTH", "MONTHLY"):
            ts_df["period"] = ts_df[date_col].dt.to_period("M").astype(str)
            period_col = "period"
            steps.append(f"Aggregate `{agg_func.upper()}({val_col})` by Monthly period")
        elif granularity in ("W", "WEEK", "WEEKLY"):
            ts_df["period"] = ts_df[date_col].dt.to_period("W").astype(str)
            period_col = "period"
            steps.append(f"Aggregate `{agg_func.upper()}({val_col})` by Weekly period")
        elif granularity in ("Q", "QUARTER", "QUARTERLY"):
            ts_df["period"] = ts_df[date_col].dt.to_period("Q").astype(str)
            period_col = "period"
            steps.append(f"Aggregate `{agg_func.upper()}({val_col})` by Quarterly period")
        elif granularity in ("Y", "YEAR", "YEARLY"):
            ts_df["period"] = ts_df[date_col].dt.to_period("Y").astype(str)
            period_col = "period"
            steps.append(f"Aggregate `{agg_func.upper()}({val_col})` by Annual period")
        else:
            ts_df["period"] = ts_df[date_col].dt.strftime("%Y-%m-%d")
            period_col = "period"
            steps.append(f"Aggregate `{agg_func.upper()}({val_col})` by Date")

        group_by_cat = resolve_column_name(inst.get("group_by") or inst.get("group_column"), ts_df)

        if group_by_cat:
            grouped = getattr(ts_df.groupby([period_col, group_by_cat])[val_col], agg_func)().reset_index()
            grouped = grouped.sort_values(by=period_col)
            steps.append(f"Break down time series by category `{group_by_cat}`")
            suggested_chart = {
                "type": "line",
                "x": period_col,
                "y": val_col,
                "color": group_by_cat,
                "title": f"{agg_func.capitalize()} of {val_col} Over Time by {group_by_cat}"
            }
        else:
            grouped = getattr(ts_df.groupby(period_col)[val_col], agg_func)().reset_index()
            grouped = grouped.sort_values(by=period_col)
            suggested_chart = {
                "type": "line",
                "x": period_col,
                "y": val_col,
                "title": f"{agg_func.capitalize()} of {val_col} Over Time"
            }

        total_sum = float(grouped[val_col].sum()) if not grouped.empty else 0.0
        if not grouped.empty and not group_by_cat:
            max_row = grouped.loc[grouped[val_col].idxmax()]
            peak_period = max_row[period_col]
            peak_val = max_row[val_col]
            formatted_res = f"Time series for {val_col} across {len(grouped)} periods (Total: {total_sum:,.2f}, Peak: {peak_period} at {peak_val:,.2f})."
        else:
            formatted_res = f"Time series trend computed for {val_col} across {len(grouped)} periods (Total: {total_sum:,.2f})."

        return AnalysisResult(
            success=True,
            operation="time_series",
            data=grouped,
            summary_table=grouped,
            formatted_result=formatted_res,
            analysis_steps=steps,
            suggested_chart=suggested_chart
        )

    def _op_distribution(self, inst: Dict[str, Any]) -> AnalysisResult:
        """Compute distribution summary for a numerical or categorical column."""
        raw_col = inst.get("column")
        col = resolve_column_name(raw_col, self.df)
        if not col:
            return AnalysisResult(
                success=False,
                operation="distribution",
                error=f"Column '{raw_col}' not found in dataset."
            )

        series = self.df[col].dropna()
        steps = [f"Calculate distribution for column `{col}`"]

        if pd.api.types.is_numeric_dtype(series):
            q25 = float(series.quantile(0.25))
            q50 = float(series.quantile(0.50))
            q75 = float(series.quantile(0.75))

            stats_df = pd.DataFrame([{
                "Column": col,
                "Count": int(len(series)),
                "Mean": round(float(series.mean()), 2),
                "Median": round(q50, 2),
                "Std": round(float(series.std()), 2) if len(series) > 1 else 0.0,
                "Min": round(float(series.min()), 2),
                "Max": round(float(series.max()), 2),
                "Q25": round(q25, 2),
                "Q75": round(q75, 2),
            }])

            return AnalysisResult(
                success=True,
                operation="distribution",
                data=stats_df,
                summary_table=stats_df,
                formatted_result=f"Distribution of '{col}': Mean = {series.mean():,.2f}, Median = {q50:,.2f}, Min = {series.min():,.2f}, Max = {series.max():,.2f}",
                analysis_steps=steps,
                suggested_chart={
                    "type": "histogram",
                    "x": col,
                    "title": f"Distribution of {col}"
                }
            )
        else:
            val_counts = series.value_counts().reset_index()
            val_counts.columns = [col, "count"]
            val_counts["percentage"] = (val_counts["count"] / len(series) * 100).round(2)

            return AnalysisResult(
                success=True,
                operation="distribution",
                data=val_counts,
                summary_table=val_counts,
                formatted_result=f"Categorical distribution for '{col}' with {len(val_counts)} unique values.",
                analysis_steps=steps,
                suggested_chart={
                    "type": "bar",
                    "x": col,
                    "y": "count",
                    "title": f"Distribution of {col}"
                }
            )

    def _op_missing_values(self, inst: Dict[str, Any]) -> AnalysisResult:
        """Analyze missing values in columns."""
        raw_col = inst.get("column")
        if raw_col:
            col = resolve_column_name(raw_col, self.df)
            if not col:
                return AnalysisResult(
                    success=False,
                    operation="missing_values",
                    error=f"Column '{raw_col}' not found."
                )
            missing = int(self.df[col].isna().sum())
            pct = round(missing / len(self.df) * 100, 2)
            summary_df = pd.DataFrame([{"Column": col, "Missing Count": missing, "Missing %": pct}])
            steps = [f"Count missing values in column `{col}`"]
            return AnalysisResult(
                success=True,
                operation="missing_values",
                data=summary_df,
                summary_table=summary_df,
                formatted_result=f"Column '{col}' has {missing:,} missing values ({pct}%).",
                analysis_steps=steps,
                suggested_chart={"type": "table", "title": f"Missing Values in {col}"}
            )

        missing_series = self.df.isna().sum()
        missing_df = pd.DataFrame({
            "Column": missing_series.index,
            "Missing Count": missing_series.values,
            "Missing %": (missing_series.values / len(self.df) * 100).round(2)
        }).sort_values(by="Missing Count", ascending=False)

        steps = ["Count missing values across all columns"]
        return AnalysisResult(
            success=True,
            operation="missing_values",
            data=missing_df,
            summary_table=missing_df,
            formatted_result=f"Total missing cells across dataset: {missing_series.sum():,}.",
            analysis_steps=steps,
            suggested_chart={
                "type": "bar",
                "x": "Column",
                "y": "Missing Count",
                "title": "Missing Values by Column"
            }
        )

    def _op_outliers(self, inst: Dict[str, Any]) -> AnalysisResult:
        """Detect and return outlier records using IQR method."""
        raw_col = inst.get("column")
        col = resolve_column_name(raw_col, self.df)
        if not col:
            return AnalysisResult(
                success=False,
                operation="outliers",
                error=f"Column '{raw_col}' not found."
            )

        series = self.df[col].dropna()
        if not pd.api.types.is_numeric_dtype(series):
            return AnalysisResult(
                success=False,
                operation="outliers",
                error=f"Column '{col}' is not numerical."
            )

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers_df = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
        steps = [
            f"Calculate IQR for `{col}` (Q1 = {q1:,.2f}, Q3 = {q3:,.2f}, IQR = {iqr:,.2f})",
            f"Identify records outside [{lower_bound:,.2f}, {upper_bound:,.2f}]"
        ]

        return AnalysisResult(
            success=True,
            operation="outliers",
            data=outliers_df,
            summary_table=outliers_df.head(20),
            formatted_result=f"Found {len(outliers_df):,} outlier records in '{col}' outside [{lower_bound:,.2f}, {upper_bound:,.2f}].",
            analysis_steps=steps,
            suggested_chart={
                "type": "box",
                "y": col,
                "title": f"Outlier Distribution for {col}"
            }
        )

    def _op_comparison(self, inst: Dict[str, Any]) -> AnalysisResult:
        """Compare a numerical metric between specific categorical groups."""
        raw_group = inst.get("group_column")
        raw_val = inst.get("value_column")

        group_col = resolve_column_name(raw_group, self.df)
        val_col = resolve_column_name(raw_val, self.df)

        if not group_col or not val_col:
            return AnalysisResult(
                success=False,
                operation="comparison",
                error="Both group_column and value_column must be valid columns."
            )

        groups = inst.get("groups", [])
        metric = str(inst.get("aggregation", inst.get("metric", "mean"))).lower()
        if metric not in ("mean", "sum", "median"):
            metric = "mean"

        sub_df = self.df.copy()
        steps = []
        if groups and isinstance(groups, list):
            sub_df = sub_df[sub_df[group_col].isin(groups)]
            steps.append(f"Filter groups: {', '.join(map(str, groups))}")

        steps.append(f"Calculate `{metric.upper()}({val_col})` grouped by `{group_col}`")
        res = getattr(sub_df.groupby(group_col)[val_col], metric)().reset_index()
        res.columns = [group_col, f"{metric}_{val_col}"]

        return AnalysisResult(
            success=True,
            operation="comparison",
            data=res,
            summary_table=res,
            formatted_result=f"Comparison of {val_col} across {group_col}.",
            analysis_steps=steps,
            suggested_chart={
                "type": "bar",
                "x": group_col,
                "y": f"{metric}_{val_col}",
                "title": f"{metric.capitalize()} {val_col} by {group_col}"
            }
        )

    def _op_ranking(self, inst: Dict[str, Any]) -> AnalysisResult:
        """Rank groups or items by an aggregated metric."""
        return self._op_groupby_aggregate(inst)

    def _op_dataset_info(self, inst: Dict[str, Any]) -> AnalysisResult:
        """Return dataset-level dimensions (exact row and column counts)."""
        total_rows = int(len(self.df))
        total_columns = int(len(self.df.columns))

        summary_df = pd.DataFrame([
            {"Dimension": "Total Rows", "Count": total_rows},
            {"Dimension": "Total Columns", "Count": total_columns}
        ])

        steps = [f"Retrieve dataset dimensions: {total_rows:,} rows × {total_columns:,} columns"]

        return AnalysisResult(
            success=True,
            operation="dataset_info",
            data={"rows": total_rows, "columns": total_columns},
            summary_table=summary_df,
            formatted_result=f"The dataset contains {total_rows:,} rows and {total_columns:,} columns.",
            analysis_steps=steps,
            suggested_chart={"type": "table", "title": "Dataset Dimensions"}
        )

    def _op_schema(self, inst: Dict[str, Any]) -> AnalysisResult:
        """Return column names, pandas data types, logical types, and non-null counts."""
        schema_rows = []
        for col in self.df.columns:
            s = self.df[col]
            if pd.api.types.is_bool_dtype(s):
                log_type = "boolean"
            elif pd.api.types.is_numeric_dtype(s):
                log_type = "numerical"
            elif pd.api.types.is_datetime64_any_dtype(s) or ("date" in str(col).lower() or "time" in str(col).lower()):
                log_type = "datetime"
            else:
                log_type = "categorical"

            schema_rows.append({
                "Column Name": str(col),
                "Data Type": str(s.dtype),
                "Logical Type": log_type,
                "Non-Null Count": int(s.count())
            })

        schema_df = pd.DataFrame(schema_rows)
        steps = [f"Inspect schema and data types across {len(self.df.columns)} columns"]

        formatted_list = [f"{r['Column Name']} ({r['Data Type']})" for r in schema_rows]
        formatted_res = f"Dataset schema ({len(schema_rows)} columns): " + ", ".join(formatted_list)

        return AnalysisResult(
            success=True,
            operation="schema",
            data=schema_df,
            summary_table=schema_df,
            formatted_result=formatted_res,
            analysis_steps=steps,
            suggested_chart={"type": "table", "title": "Column Schema & Data Types"}
        )
