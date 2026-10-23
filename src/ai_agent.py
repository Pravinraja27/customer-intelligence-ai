"""
ai_agent.py
Built-in analytics assistant with optional OpenAI LLM integration.
"""
import os
import textwrap
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
_openai_available = False

if OPENAI_API_KEY and OPENAI_API_KEY != "your_api_key_here":
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        _openai_available = True
    except ImportError:
        _openai_available = False


# ────────────────────────────────────────────────────────────────
# Tool functions — used by both built-in and LLM modes
# ────────────────────────────────────────────────────────────────

def get_dataset_summary(state: dict) -> str:
    df = state.get("df")
    if df is None:
        return "No dataset is currently loaded."
    from src.analytics import summarize_dataset
    cluster_summary = state.get("cluster_summary")
    return summarize_dataset(df, cluster_summary)


def get_dashboard_kpis(state: dict) -> str:
    kpis = state.get("kpis")
    if not kpis:
        return "KPI data is not available."
    lines = [f"Total Customers: {kpis.get('total_customers', 'N/A')}"]
    mapping = {
        "avg_annual_income": "Average Annual Income",
        "avg_spending_score": "Average Spending Score",
        "avg_order_value": "Average Order Value",
        "avg_purchase_freq": "Average Purchase Frequency",
        "avg_satisfaction": "Average Customer Satisfaction",
        "avg_recency_days": "Average Recency Days",
    }
    for key, label in mapping.items():
        val = kpis.get(key)
        if val:
            lines.append(f"{label}: {val}")
    return "\n".join(lines)


def get_cluster_summary_text(state: dict) -> str:
    cs = state.get("cluster_summary")
    if not cs:
        return "No clustering results available. Please run segmentation first."
    lines = [f"Number of customer segments: {cs['n_clusters']}"]
    for seg in cs.get("segments", []):
        profile_str = ", ".join(f"{k}={v}" for k, v in seg["profile"].items())
        lines.append(
            f"  Segment '{seg['label']}': {seg['size']} customers ({seg['pct']:.1f}%) — {profile_str}"
        )
    return "\n".join(lines)


def get_model_metrics_text(state: dict) -> str:
    pm = state.get("pred_result")
    if not pm:
        return "No prediction model has been trained yet."
    from src.prediction import get_model_metrics_summary
    return get_model_metrics_summary(pm)


def get_top_customer_segments(state: dict) -> str:
    cs = state.get("cluster_summary")
    if not cs:
        return "Segmentation not yet performed."
    segs = sorted(cs.get("segments", []), key=lambda x: x.get("pct", 0), reverse=True)
    lines = ["Top customer segments by size:"]
    for s in segs:
        lines.append(f"  - {s['label']}: {s['size']} customers ({s['pct']:.1f}%)")
    return "\n".join(lines)


def get_data_quality_report_text(state: dict) -> str:
    dq = state.get("data_quality")
    if not dq:
        return "Data quality report not available."
    lines = [
        f"Total records: {dq['total_records']}",
        f"Total columns: {dq['total_columns']}",
        f"Missing values: {dq['missing_values_total']}",
        f"Duplicate rows: {dq['duplicate_rows']}",
        f"Memory usage: {dq['memory_mb']} MB",
    ]
    if dq.get("missing_by_column"):
        lines.append("Columns with missing values: " +
                     ", ".join(f"{k}={v}" for k, v in dq["missing_by_column"].items()))
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────
# Built-in rule-based assistant
# ────────────────────────────────────────────────────────────────

_BUILTIN_RULES = [
    (
        ["summarize", "dataset", "data summary", "overview", "describe the data"],
        lambda s: (
            "📊 **Dataset Overview**\n\n" + get_dataset_summary(s) +
            "\n\n**KPIs:**\n" + get_dashboard_kpis(s)
        ),
    ),
    (
        ["segment", "cluster", "segmentation", "groups", "customer group"],
        lambda s: (
            "🎯 **Customer Segments**\n\n" + get_cluster_summary_text(s) +
            "\n\n" + get_top_customer_segments(s)
        ),
    ),
    (
        ["highest spending", "high spend", "top spender", "most valuable"],
        lambda s: _top_spender_response(s),
    ),
    (
        ["pattern", "insight", "trend", "observation"],
        lambda s: _pattern_response(s),
    ),
    (
        ["retention", "churn", "strategy", "recommend", "improve"],
        lambda s: _retention_response(s),
    ),
    (
        ["predict", "model", "r2", "mae", "accuracy", "r²", "forecast"],
        lambda s: (
            "🤖 **Prediction Model Results**\n\n" + get_model_metrics_text(s) +
            "\n\n*Note: Predictions are based on training data and are intended for demonstration only.*"
        ),
    ),
    (
        ["quality", "missing", "duplicate", "data quality"],
        lambda s: "🔍 **Data Quality Report**\n\n" + get_data_quality_report_text(s),
    ),
    (
        ["kpi", "metric", "key performance", "average income", "spending score"],
        lambda s: "📈 **Dashboard KPIs**\n\n" + get_dashboard_kpis(s),
    ),
    (
        ["presentation", "internship", "summary", "demo", "report"],
        lambda s: _presentation_summary(s),
    ),
]


def _top_spender_response(state: dict) -> str:
    cs = state.get("cluster_summary")
    if not cs:
        return "Please run the Customer Segmentation first to identify spending patterns."
    segs = cs.get("segments", [])
    ranked = sorted(
        segs,
        key=lambda x: x["profile"].get("SpendingScore", x["profile"].get("AverageOrderValue", 0)),
        reverse=True,
    )
    if not ranked:
        return "No segment data available."
    top = ranked[0]
    profile_str = ", ".join(f"{k}={v}" for k, v in top["profile"].items())
    return (
        f"💰 **Highest-Spending Segment: {top['label']}**\n\n"
        f"- Size: {top['size']} customers ({top['pct']:.1f}%)\n"
        f"- Profile: {profile_str}\n\n"
        f"This segment has the highest average spending score among all identified clusters."
    )


def _pattern_response(state: dict) -> str:
    summary = get_dataset_summary(state)
    kpis = get_dashboard_kpis(state)
    cs_text = get_cluster_summary_text(state)
    return (
        "🔍 **Key Patterns in the Data**\n\n"
        + summary + "\n\n"
        + "**Segments identified:**\n" + cs_text + "\n\n"
        "**Observations:** Customers vary significantly in spending behaviour, income, "
        "and purchase frequency. Segmentation reveals distinct groups that require "
        "tailored marketing strategies."
    )


def _retention_response(state: dict) -> str:
    cs = state.get("cluster_summary")
    strategies = [
        "📌 **Customer Retention Strategies Based on Segment Analysis**\n",
    ]
    if cs:
        for seg in cs.get("segments", []):
            label = seg["label"].lower()
            name = seg["label"]
            if "high" in label or "vip" in label or "premium" in label:
                strategies.append(
                    f"- **{name}** → Offer exclusive loyalty rewards, early access to new products, "
                    "and personalised account managers."
                )
            elif "lapsed" in label or "churn" in label or "at-risk" in label:
                strategies.append(
                    f"- **{name}** → Re-engage with win-back campaigns, targeted discounts, "
                    "and surveys to understand drop-off reasons."
                )
            elif "budget" in label or "low" in label:
                strategies.append(
                    f"- **{name}** → Highlight value deals, bundle offers, and referral bonuses "
                    "to increase average order value."
                )
            else:
                strategies.append(
                    f"- **{name}** → Nurture with regular engagement emails, satisfaction surveys, "
                    "and loyalty points."
                )
    else:
        strategies.append(
            "Run the Customer Segmentation module first to get personalised recommendations. "
            "General advice: personalise outreach, reward loyal customers, and re-engage lapsed ones."
        )
    return "\n".join(strategies)


def _presentation_summary(state: dict) -> str:
    kpis = get_dashboard_kpis(state)
    cs_text = get_cluster_summary_text(state)
    model_text = get_model_metrics_text(state)
    return textwrap.dedent(f"""
    🎓 **Internship Presentation Summary — Customer Intelligence AI**

    **Project:** Intelligent Customer Segmentation & Predictive Analytics Dashboard
    **Organisation:** BICS Global | Role: Data Analyst Intern

    **Dataset KPIs:**
    {kpis}

    **Segmentation Results:**
    {cs_text}

    **Predictive Model:**
    {model_text}

    **Key Takeaways:**
    - The application uses K-Means clustering to group customers into meaningful segments.
    - A RandomForest regression model predicts customer spending scores.
    - An AI assistant provides natural language insights grounded in calculated data.
    - All results are reproducible and based on the active dataset.

    *This summary is generated from live application data and is suitable for presentation.*
    """).strip()


# ────────────────────────────────────────────────────────────────
# Main respond function
# ────────────────────────────────────────────────────────────────

def respond(user_message: str, state: dict, chat_history: list[dict]) -> tuple[str, str]:
    """
    Generate a response for the user's message.
    Returns (response_text, mode) where mode is 'builtin' or 'llm'.
    """
    lower = user_message.lower().strip()

    if _openai_available:
        try:
            response = _llm_respond(user_message, state, chat_history)
            return response, "llm"
        except Exception as e:
            # Fall through to built-in
            pass

    # Built-in rule-based routing
    for keywords, handler in _BUILTIN_RULES:
        if any(kw in lower for kw in keywords):
            return handler(state), "builtin"

    # Generic fallback
    return (
        "I'm the Customer Intelligence Assistant 🤖\n\n"
        "I can help you with:\n"
        "- Dataset summary and KPIs\n"
        "- Customer segment details\n"
        "- Predictive model results\n"
        "- Data quality insights\n"
        "- Retention strategies\n\n"
        "Try one of the suggested questions below, or ask me something specific about your data!"
    ), "builtin"


def _llm_respond(user_message: str, state: dict, chat_history: list[dict]) -> str:
    """
    Generate a response using the OpenAI API, grounded in calculated analytics context.
    """
    import openai

    # Build context block
    context_parts = [
        "=== DATASET SUMMARY ===\n" + get_dataset_summary(state),
        "=== KPIs ===\n" + get_dashboard_kpis(state),
        "=== CLUSTER SUMMARY ===\n" + get_cluster_summary_text(state),
        "=== MODEL METRICS ===\n" + get_model_metrics_text(state),
        "=== DATA QUALITY ===\n" + get_data_quality_report_text(state),
    ]
    context = "\n\n".join(context_parts)

    system_prompt = (
        "You are 'Customer Intelligence Agent', an analytics assistant for the "
        "Customer Intelligence AI dashboard at BICS Global internship project. "
        "You help users understand customer segmentation, predictive analytics, and business insights.\n\n"
        "IMPORTANT RULES:\n"
        "1. Only use the numerical values provided in the context below. Do NOT invent statistics.\n"
        "2. If information is unavailable, say so clearly.\n"
        "3. Be concise, professional, and helpful.\n"
        "4. Use markdown formatting for readability.\n\n"
        f"=== LIVE ANALYTICS CONTEXT ===\n{context}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    # Include last 6 messages of history
    for entry in chat_history[-6:]:
        messages.append({"role": entry["role"], "content": entry["content"]})
    messages.append({"role": "user", "content": user_message})

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=600,
        temperature=0.4,
    )
    return res.choices[0].message.content.strip()


SUGGESTED_QUESTIONS = [
    "Summarize the customer dataset.",
    "Explain the customer segments.",
    "Which customer segment has the highest spending?",
    "What patterns can you identify in the data?",
    "What strategies could improve customer retention?",
    "Explain the predictive model results.",
    "Give me a summary for my internship presentation.",
]
