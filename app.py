"""AI Analytics Engineer - Streamlit Application.

A portfolio-grade AI-powered data analytics workspace.
The LLM is the analyst's brain. Python/Pandas is the calculator.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Ensure root directory is on Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.analyzer import AnalyticsEngine, AnalysisResult
from src.llm import explain_result_with_llm, parse_question_with_llm
from src.profiler import profile_dataset
from src.visualizer import (
    create_chart_from_result,
    plot_bar_chart,
    plot_box,
    plot_correlation_heatmap,
    plot_histogram,
    plot_line_chart,
    plot_scatter,
)

# Load environment variables
load_dotenv()

# Streamlit Page Config
st.set_page_config(
    page_title="AI Analytics Engineer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium Custom CSS
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

code, pre, [class*="stCode"] {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Header styling */
.hero-container {
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
}

.hero-title {
    font-size: 2.25rem;
    font-weight: 800;
    color: #FFFFFF;
    margin: 0;
    letter-spacing: -0.02em;
}

.hero-title span {
    background: linear-gradient(90deg, #60A5FA 0%, #34D399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #94A3B8;
    margin-top: 0.5rem;
    margin-bottom: 0;
}

/* Metric card */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px -2px rgba(0,0,0,0.08);
}

.kpi-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.kpi-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #0F172A;
    margin-top: 0.25rem;
}

/* Answer Callout Card */
.answer-box {
    background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
    border-left: 5px solid #3B82F6;
    border-radius: 10px;
    padding: 1.5rem 1.75rem;
    margin-top: 1rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}

.answer-header {
    font-size: 0.85rem;
    font-weight: 700;
    color: #2563EB;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.5rem;
}

.answer-text {
    font-size: 1.2rem;
    font-weight: 600;
    color: #0F172A;
    line-height: 1.5;
}

/* Insight Cards */
.insight-card {
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.9rem;
    border: 1px solid #E2E8F0;
    background-color: #FFFFFF;
}

.insight-info { border-left: 4px solid #3B82F6; }
.insight-warning { border-left: 4px solid #F59E0B; background-color: #FFFDF7; }
.insight-high { border-left: 4px solid #10B981; }

.insight-title {
    font-weight: 700;
    font-size: 0.98rem;
    color: #1E293B;
    margin-bottom: 0.25rem;
}

.insight-desc {
    font-size: 0.9rem;
    color: #475569;
    line-height: 1.45;
}

/* Steps badge */
.step-badge {
    display: inline-block;
    background: #E0E7FF;
    color: #3730A3;
    padding: 0.3rem 0.75rem;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
}

/* Quick question pills */
.stButton > button {
    border-radius: 8px;
    font-weight: 500;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_csv_file(file_content, filename: str) -> pd.DataFrame:
    """Robust CSV loader with fallback encodings and error handlers."""
    try:
        return pd.read_csv(file_content)
    except UnicodeDecodeError:
        file_content.seek(0)
        return pd.read_csv(file_content, encoding="latin1")
    except Exception as e:
        raise ValueError(f"Error parsing CSV '{filename}': {str(e)}")


def load_sample_dataset() -> pd.DataFrame:
    """Load the built-in realistic e-commerce sample dataset."""
    sample_path = Path(__file__).parent / "data" / "sample.csv"
    if sample_path.exists():
        return pd.read_csv(sample_path)
    raise FileNotFoundError("Sample dataset not found at data/sample.csv")


def init_session_state():
    """Initialize session state variables."""
    if "df" not in st.session_state:
        st.session_state.df = None
    if "df_name" not in st.session_state:
        st.session_state.df_name = None
    if "profile" not in st.session_state:
        st.session_state.profile = None
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if "current_question" not in st.session_state:
        st.session_state.current_question = ""


init_session_state()

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/isometric/96/combo-chart.png", width=54)
    st.title("AI Analytics Engine")
    st.caption("Deterministic calculations powered by Pandas & LLM logic.")
    
    st.markdown("---")
    st.subheader("📁 Data Ingestion")
    
    uploaded_file = st.file_uploader(
        "Upload CSV Dataset",
        type=["csv"],
        help="Upload any structured CSV file to begin analysis.",
        key="csv_uploader"
    )
    
    col_demo, col_clear = st.columns([1, 1])
    with col_demo:
        if st.button("🚀 Load Sample", use_container_width=True, help="Load built-in 2025 E-Commerce dataset"):
            sample_data = load_sample_dataset()
            st.session_state.df = sample_data
            st.session_state.df_name = "sample_ecommerce_2025.csv"
            st.session_state.profile = profile_dataset(sample_data)
            st.rerun()
            
    with col_clear:
        if st.button("🗑️ Clear Data", use_container_width=True):
            st.session_state.df = None
            st.session_state.df_name = None
            st.session_state.profile = None
            st.session_state.query_history = []
            st.rerun()

    # Handle file upload event
    if uploaded_file is not None:
        if st.session_state.df_name != uploaded_file.name:
            try:
                df_loaded = load_csv_file(uploaded_file, uploaded_file.name)
                if df_loaded.empty:
                    st.error("Uploaded CSV file is empty.")
                else:
                    st.session_state.df = df_loaded
                    st.session_state.df_name = uploaded_file.name
                    st.session_state.profile = profile_dataset(df_loaded)
                    st.toast(f"Loaded '{uploaded_file.name}' successfully!", icon="✅")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to read file: {e}")

    st.markdown("---")
    st.subheader("⚙️ LLM Configuration")
    
    env_api_key = os.getenv("OPENAI_API_KEY", "")
    api_key_input = st.text_input(
        "OpenAI API Key",
        value=env_api_key if env_api_key != "your_openai_api_key_here" else "",
        type="password",
        placeholder="sk-...",
        help="Provide your OpenAI API key. If left blank, the engine operates in Deterministic Heuristic Mode.",
    )
    
    selected_model = st.selectbox(
        "Model",
        options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0,
    )
    
    active_api_key = api_key_input.strip() if api_key_input.strip() else None
    if active_api_key:
        st.success("🤖 OpenAI API Active", icon="✨")
    else:
        st.info("⚡ Heuristic Calculator Mode", icon="ℹ️")

    # Dataset quick stats in sidebar
    if st.session_state.df is not None:
        st.markdown("---")
        st.subheader("📋 Active Dataset")
        st.write(f"**File:** `{st.session_state.df_name}`")
        st.write(f"**Rows:** `{len(st.session_state.df):,}`")
        st.write(f"**Columns:** `{len(st.session_state.df.columns)}`")
        st.write(f"**Memory:** `{st.session_state.df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB`")


# ==========================================
# MAIN PAGE HERO HEADER
# ==========================================
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">AI <span>Analytics Engineer</span></h1>
    <p class="hero-subtitle">Ask questions. Get verified answers. Understand your data with mathematical precision.</p>
</div>
""", unsafe_allow_html=True)


# If no dataset loaded, display welcome & getting started guide
if st.session_state.df is None:
    st.info("👋 Welcome! Upload a CSV file in the sidebar or click **'Load Sample'** to instantly explore the analytics workspace.", icon="💡")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.markdown("""
        ### 🧠 LLM as the Brain
        The LLM translates questions into structured analytical specifications. Arbitrary code execution is strictly prohibited.
        """)
    with col_f2:
        st.markdown("""
        ### 🔢 Pandas as Calculator
        100% deterministic calculations executed natively in Python. The system never invents or hallucinates numerical results.
        """)
    with col_f3:
        st.markdown("""
        ### 📊 Visualizations & Insights
        Generates automatic dataset profiles, correlation matrices, IQR outlier scans, and dynamic Plotly visualizations.
        """)
    st.stop()


df: pd.DataFrame = st.session_state.df
profile: dict = st.session_state.profile

# ==========================================
# TABS NAVIGATION
# ==========================================
tab_query, tab_overview, tab_insights, tab_visualizer = st.tabs([
    "💬 Ask Your Data",
    "📊 Dataset Profile",
    "💡 Automated Insights",
    "📈 Interactive Visualizer"
])


# ------------------------------------------
# TAB 1: ASK YOUR DATA
# ------------------------------------------
with tab_query:
    st.markdown("### 💬 Natural Language Data Querying")
    st.caption("Ask questions in plain English. The AI formulates the analytical instruction, Pandas executes the calculation, and verified results are presented.")
    
    # Suggested Question Pills
    st.markdown("**💡 Example Questions:**")
    quick_questions = [
        "What is the average revenue?",
        "Which product generated the most revenue?",
        "What is the total revenue by category?",
        "How did revenue change over time?",
        "Is advertising_spend related to revenue?",
        "Which region has the highest average customer_rating?",
        "Show distribution of customer_rating",
        "What are the top 5 products by units_sold?",
    ]
    
    cols = st.columns(4)
    for i, q in enumerate(quick_questions):
        with cols[i % 4]:
            if st.button(q, key=f"quick_q_{i}", use_container_width=True):
                st.session_state.current_question = q
                st.rerun()

    # Query Input
    query_form = st.form(key="query_form", clear_on_submit=False)
    user_query = query_form.text_input(
        "Enter your question about this dataset:",
        value=st.session_state.current_question,
        placeholder="e.g., Which category has the highest total sales?",
    )
    submit_button = query_form.form_submit_button("🔍 Run Analysis", use_container_width=True)

    if submit_button and user_query.strip():
        question = user_query.strip()
        st.session_state.current_question = question
        
        with st.spinner("🤖 Formulating analytical instruction and executing calculation..."):
            # 1. LLM Translation
            instruction, is_llm, parse_error = parse_question_with_llm(
                question=question,
                df=df,
                api_key=active_api_key,
                model=selected_model
            )
            
            # 2. Python/Pandas Execution
            engine = AnalyticsEngine(df)
            result: AnalysisResult = engine.execute(instruction)
            
            # 3. Verified Natural Language Explanation
            explanation = explain_result_with_llm(
                question=question,
                instruction=instruction,
                result=result,
                api_key=active_api_key,
                model=selected_model
            )
            
            # Save to history
            st.session_state.query_history.insert(0, {
                "question": question,
                "instruction": instruction,
                "result": result,
                "explanation": explanation,
                "is_llm": is_llm,
                "parse_error": parse_error
            })

    # Render Latest Query Result
    if st.session_state.query_history:
        latest = st.session_state.query_history[0]
        res: AnalysisResult = latest["result"]
        inst = latest["instruction"]
        
        st.markdown("---")
        st.markdown(f"#### 🎯 Query: *\"{latest['question']}\"*")
        
        if res.success:
            # 1. Natural Language Answer Box
            st.markdown(f"""
            <div class="answer-box">
                <div class="answer-header">Verified AI Answer</div>
                <div class="answer-text">{latest['explanation']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # 2. Transparency Section: Analysis Performed
            st.markdown("##### 🔍 Analysis Performed")
            steps_html = "".join([f'<span class="step-badge">✓ {step}</span>' for step in res.analysis_steps])
            st.markdown(steps_html, unsafe_allow_html=True)
            
            # Collapsible Structured Instruction
            with st.expander("🛠️ View Structured Analytical Instruction (JSON)", expanded=False):
                col_inst, col_meta = st.columns([2, 1])
                with col_inst:
                    st.json(inst)
                with col_meta:
                    st.write(f"**Parser Mode:** {'OpenAI LLM' if latest['is_llm'] else 'Deterministic Rule Engine'}")
                    st.write(f"**Operation:** `{res.operation}`")
                    if latest["parse_error"]:
                        st.caption(f"Note: {latest['parse_error']}")

            # 3. Supporting Visualizations & Data Table
            col_chart, col_data = st.columns([3, 2] if res.summary_table is not None and not res.summary_table.empty else [1, 0.01])
            
            chart = create_chart_from_result(res.suggested_chart, res.data, df)
            
            with col_chart:
                if chart:
                    st.markdown("##### 📈 Visual Evidence")
                    st.plotly_chart(chart, use_container_width=True)
                elif isinstance(res.data, (int, float, np.number)):
                    st.markdown("##### 📌 Computed Metric")
                    st.metric(label=res.suggested_chart.get("label", "Value") if res.suggested_chart else "Result", value=f"{res.data:,.2f}")
                    
            with col_data:
                if res.summary_table is not None and not res.summary_table.empty:
                    st.markdown("##### 📋 Result Table")
                    st.dataframe(res.summary_table, use_container_width=True, hide_index=True)
                    
        else:
            st.error(f"❌ Analysis Failed: {res.error}")
            with st.expander("View Attempted Instruction"):
                st.json(inst)


# ------------------------------------------
# TAB 2: DATASET OVERVIEW & PROFILE
# ------------------------------------------
with tab_overview:
    st.markdown("### 📊 Dataset Overview & Statistical Profile")
    
    # KPI Overview Cards
    gen = profile["general"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Rows</div>
            <div class="kpi-value">{gen['total_rows']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Columns</div>
            <div class="kpi-value">{gen['total_columns']}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Duplicate Rows</div>
            <div class="kpi-value">{gen['duplicate_rows']:,} <span style="font-size:0.9rem; color:#64748B;">({gen['duplicate_percentage']}%)</span></div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Missing Cells</div>
            <div class="kpi-value">{gen['total_missing_cells']:,} <span style="font-size:0.9rem; color:#64748B;">({gen['total_missing_percentage']}%)</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Preview
    st.markdown("#### 🔍 Data Preview")
    preview_rows = st.slider("Rows to preview", min_value=5, max_value=min(100, len(df)), value=min(10, len(df)))
    st.dataframe(df.head(preview_rows), use_container_width=True)

    # Column Schema Profile
    st.markdown("#### 📐 Column Schema & Data Types")
    cols_df = pd.DataFrame(profile["columns"])
    cols_df["sample_values"] = cols_df["sample_values"].apply(lambda v: ", ".join(v[:3]))
    cols_df = cols_df.rename(columns={
        "name": "Column Name",
        "logical_type": "Logical Type",
        "raw_dtype": "Dtype",
        "unique_count": "Unique Count",
        "missing_count": "Missing Count",
        "missing_percentage": "Missing %",
        "sample_values": "Sample Values"
    })
    st.dataframe(cols_df, use_container_width=True, hide_index=True)

    # Numerical Summary
    if profile["numerical"]:
        st.markdown("#### 🔢 Numerical Columns Summary")
        num_summary_df = pd.DataFrame(profile["numerical"]).T.reset_index()
        num_summary_df = num_summary_df.rename(columns={"index": "Column"})
        st.dataframe(num_summary_df, use_container_width=True, hide_index=True)

    # Categorical Summary
    if profile["categorical"]:
        st.markdown("#### 🔤 Categorical Columns Breakdown")
        cat_cols_list = list(profile["categorical"].keys())
        selected_cat = st.selectbox("Select Categorical Column to Inspect:", cat_cols_list)
        if selected_cat:
            cat_info = profile["categorical"][selected_cat]
            top_vals_df = pd.DataFrame(cat_info["top_values"])
            col_bar, col_tbl = st.columns([3, 2])
            with col_bar:
                fig_cat = plot_bar_chart(top_vals_df, x="value", y="count", title=f"Top Frequencies for '{selected_cat}'")
                st.plotly_chart(fig_cat, use_container_width=True)
            with col_tbl:
                st.dataframe(top_vals_df, use_container_width=True, hide_index=True)

    # Outlier Analysis (IQR)
    if profile["outliers"]:
        st.markdown("#### 🚨 Outlier Detection (IQR Method)")
        outlier_records = []
        for col_name, o_data in profile["outliers"].items():
            outlier_records.append({
                "Column": col_name,
                "Outliers Found": o_data["outlier_count"],
                "Outlier %": f"{o_data['outlier_percentage']}%",
                "Lower Bound": o_data["lower_bound"],
                "Upper Bound": o_data["upper_bound"],
                "IQR": o_data["iqr"],
            })
        outlier_table = pd.DataFrame(outlier_records)
        st.dataframe(outlier_table, use_container_width=True, hide_index=True)


# ------------------------------------------
# TAB 3: AUTOMATED INSIGHTS
# ------------------------------------------
with tab_insights:
    st.markdown("### 💡 Automated Insights")
    st.caption("Deterministic findings extracted directly from Pandas calculations. No hallucinations.")
    
    insights = profile["insights"]
    if not insights:
        st.info("No notable anomalies or insights identified for this dataset.")
    else:
        for ins in insights:
            badge_class = "insight-warning" if ins["importance"] == "warning" else ("insight-high" if ins["importance"] == "high" else "insight-info")
            badge_icon = "⚠️" if ins["importance"] == "warning" else ("🔥" if ins["importance"] == "high" else "ℹ️")
            st.markdown(f"""
            <div class="insight-card {badge_class}">
                <div class="insight-title">{badge_icon} [{ins['category']}] {ins['title']}</div>
                <div class="insight-desc">{ins['description']}</div>
            </div>
            """, unsafe_allow_html=True)


# ------------------------------------------
# TAB 4: INTERACTIVE VISUALIZER
# ------------------------------------------
with tab_visualizer:
    st.markdown("### 📈 Interactive Visualizer")
    st.caption("Explore custom charts, distributions, correlations, and multi-variable relationships.")
    
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    cat_cols = [c for c in df.columns if c not in num_cols]
    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower() or pd.api.types.is_datetime64_any_dtype(df[c])]
    
    viz_type = st.radio(
        "Select Visualization Mode:",
        options=["🔥 Correlation Heatmap", "📊 Distribution & Histograms", "📈 Time Series Trends", "🔵 Scatter & Correlation", "📦 Box & Outlier Plots"],
        horizontal=True
    )
    
    if viz_type == "🔥 Correlation Heatmap":
        if profile["correlations"] is not None:
            fig_corr = plot_correlation_heatmap(profile["correlations"])
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("At least 2 numerical columns are required to generate a correlation heatmap.")
            
    elif viz_type == "📊 Distribution & Histograms":
        col_select = st.selectbox("Select Numerical Column for Distribution:", num_cols if num_cols else df.columns)
        if col_select:
            bins = st.slider("Histogram Bins", min_value=5, max_value=60, value=25)
            fig_hist = plot_histogram(df, column=col_select, nbins=bins)
            st.plotly_chart(fig_hist, use_container_width=True)
            
    elif viz_type == "📈 Time Series Trends":
        if date_cols and num_cols:
            c1, c2, c3 = st.columns(3)
            with c1:
                selected_date = st.selectbox("Date Column:", date_cols)
            with c2:
                selected_val = st.selectbox("Metric Column:", num_cols)
            with c3:
                selected_cat = st.selectbox("Group By Category (Optional):", ["None"] + cat_cols)
                
            agg_type = st.selectbox("Aggregation:", ["sum", "mean", "median", "max", "min"])
            
            engine = AnalyticsEngine(df)
            ts_inst = {
                "operation": "time_series",
                "date_column": selected_date,
                "value_column": selected_val,
                "aggregation": agg_type,
                "time_granularity": "M",
                "group_by": selected_cat if selected_cat != "None" else None
            }
            ts_res = engine.execute(ts_inst)
            if ts_res.success:
                ts_chart = create_chart_from_result(ts_res.suggested_chart, ts_res.data, df)
                if ts_chart:
                    st.plotly_chart(ts_chart, use_container_width=True)
            else:
                st.error(ts_res.error)
        else:
            st.warning("Time series visualization requires at least one date/time column and one numerical column.")

    elif viz_type == "🔵 Scatter & Correlation":
        if len(num_cols) >= 2:
            c1, c2, c3 = st.columns(3)
            with c1:
                x_axis = st.selectbox("X Axis:", num_cols, index=0)
            with c2:
                y_axis = st.selectbox("Y Axis:", num_cols, index=min(1, len(num_cols) - 1))
            with c3:
                color_axis = st.selectbox("Color By (Optional):", ["None"] + cat_cols)
                
            fig_scat = plot_scatter(df, x=x_axis, y=y_axis, color=color_axis if color_axis != "None" else None)
            st.plotly_chart(fig_scat, use_container_width=True)
        else:
            st.warning("At least 2 numerical columns are required for a scatter plot.")

    elif viz_type == "📦 Box & Outlier Plots":
        if num_cols:
            c1, c2 = st.columns(2)
            with c1:
                box_val = st.selectbox("Numerical Value:", num_cols)
            with c2:
                box_cat = st.selectbox("Group By (Optional):", ["None"] + cat_cols)
                
            fig_box = plot_box(df, y=box_val, x=box_cat if box_cat != "None" else None)
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning("Numerical columns required for box plots.")
