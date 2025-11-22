import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

# -----------------------------
# Config: API + model + data
# -----------------------------
load_dotenv()  # loads OPENAI_API_KEY from .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

DATA_PATH = "data.xlsx"  # assumes data.xlsx is in the same folder as this script


# -----------------------------
# Helpers: data loading & parsing
# -----------------------------
@st.cache_data
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_excel(path)
    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]
    return df


def detect_period_columns(df: pd.DataFrame, base_name: str):
    """
    Find all columns that start with base_name (e.g. 'Spend ', 'Support ', 'CPP ')
    and are NOT the 'Change' column. Return (prev_col, curr_col) sorted by name.

    Example: for ['Spend Q324','Spend Q325','Spend Change'] and base_name='Spend '
    -> ('Spend Q324', 'Spend Q325')

    This works for any naming like:
      - Spend Q324 / Spend Q325
      - Spend FY24 / Spend FY25
      - Spend H1FY24 / Spend H1FY25
    as long as they consistently start with 'Spend ', 'Support ', 'CPP ' and
    have a separate '... Change' column.
    """
    candidates = [c for c in df.columns if c.startswith(base_name) and "Change" not in c]
    if len(candidates) < 2:
        raise ValueError(f"Not enough period columns found for {base_name!r}: {candidates}")

    candidates = sorted(candidates)  # lexicographic sort
    prev_col, curr_col = candidates[-2], candidates[-1]
    return prev_col, curr_col


def parse_pct_value(val) -> float:
    """
    Convert a cell value into a % float like -30.1, 23.8 etc.
    Handles:
      - " -30.1%"  (string)
      - -0.301     (fraction)
      - 30.1       (already percent)
    """
    if isinstance(val, str):
        s = val.replace("%", "").replace(",", "").strip()
        if s == "":
            return 0.0
        try:
            return float(s)
        except ValueError:
            return 0.0

    # numeric
    try:
        v = float(val)
    except (TypeError, ValueError):
        return 0.0

    # If it looks like a fraction (-0.3), convert to %
    if -2.0 < v < 2.0:
        return v * 100.0
    return v


def fmt_pct(p: float) -> str:
    sign = "+" if p > 0 else ""
    return f"{sign}{p:.1f}%"


def classify_change(p: float, name: str) -> str:
    """
    Produce a qualitative description for a % change.
    Example: -61% support -> "Support dropped sharply (~-61%)"
    """
    abs_p = abs(p)
    if p > 0:
        direction = "increased"
    elif p < 0:
        direction = "decreased"
    else:
        direction = "stayed broadly flat"

    if abs_p < 5:
        intensity = "stayed broadly flat"
    elif abs_p < 20:
        intensity = "changed slightly"
    elif abs_p < 50:
        intensity = "changed meaningfully"
    else:
        intensity = "changed sharply"

    if p == 0:
        return f"{name} {intensity} (≈0%)."
    return f"{name} {intensity}, {direction} by around {fmt_pct(p)}."


def infer_effectiveness_and_roas(spend_pct: float, support_pct: float, cpp_pct: float):
    """
    Rough directional logic for Effectiveness and ROAS.
    We don't have incremental volume, but we can infer likely direction.

    Effectiveness ~ Incremental Volume / Support
      -> driven primarily by support & CPP (cost per unit support)

    ROAS ~ Incremental Volume / Spend
      -> depends on both effectiveness and spend.
    """
    # Effectiveness: driven primarily by support & CPP
    eff_dir = "mixed or uncertain"
    eff_reason = []

    if support_pct > 10 and cpp_pct < -10:
        eff_dir = "likely improved significantly"
        eff_reason.append(
            "Support has grown strongly while CPP has reduced materially, "
            "so the channel is delivering a lot more volume per unit of spend."
        )
    elif support_pct > 0 and cpp_pct < 0:
        eff_dir = "likely improved moderately"
        eff_reason.append(
            "Support is up and CPP has also eased, pointing to more volume per unit of spend."
        )
    elif support_pct < -10 and cpp_pct > 10:
        eff_dir = "likely deteriorated significantly"
        eff_reason.append(
            "Support has collapsed while CPP has inflated, meaning far fewer impressions/GRPs "
            "for every rupee/dollar invested."
        )
    elif support_pct <= 0 and cpp_pct > 0:
        eff_dir = "likely deteriorated"
        eff_reason.append(
            "Support is not growing and CPP is higher, implying weaker efficiency."
        )
    else:
        eff_reason.append(
            "The movements in support and CPP offset each other, so net effectiveness is ambiguous."
        )

    # ROAS: depends on both effectiveness direction and Spend movement
    roas_dir = "mixed or uncertain"
    roas_reason = []

    if "improved" in eff_dir:
        if spend_pct <= 0:
            roas_dir = "likely improved strongly"
            roas_reason.append(
                "Effectiveness is up while spend is flat or lower, so every rupee/dollar "
                "should be generating more incremental volume."
            )
        else:
            roas_dir = "likely improved"
            roas_reason.append(
                "Effectiveness is up and spend is also higher, so incremental volume should grow "
                "faster than investment, supporting better ROAS unless diminishing returns kick in."
            )
    elif "deteriorated" in eff_dir:
        if spend_pct >= 0:
            roas_dir = "likely deteriorated significantly"
            roas_reason.append(
                "Effectiveness is down and spend is flat or higher, so incremental volume per unit spend "
                "should be under clear pressure."
            )
        else:
            roas_dir = "likely deteriorated"
            roas_reason.append(
                "Effectiveness is down but spend has been cut, so ROAS may still be weaker "
                "versus last year, although absolute losses are somewhat contained."
            )
    else:
        roas_reason.append(
            "With ambiguous effectiveness trends and changing spend, ROAS direction is not clear-cut."
        )

    eff_text = (
        f"Effectiveness (incremental volume per unit of support) is {eff_dir}. "
        + " ".join(eff_reason)
    )
    roas_text = (
        f"ROAS (incremental volume per unit of spend) is {roas_dir}. "
        + " ".join(roas_reason)
    )

    return eff_text, roas_text


def build_prompt(
    business_vertical: str,
    variable: str,
    row: pd.Series,
    spend_prev_col: str,
    spend_curr_col: str,
    support_prev_col: str,
    support_curr_col: str,
    cpp_prev_col: str,
    cpp_curr_col: str,
    spend_pct: float,
    support_pct: float,
    cpp_pct: float,
    eff_text: str,
    roas_text: str,
) -> list:
    """
    Build messages list for OpenAI chat completion.
    Uses generic "previous vs current" period names instead of hard-coded quarters.
    """

    system_msg = {
        "role": "system",
        "content": """
You are a senior marketing and media strategist supporting Marketing Mix Modeling (MMM).
You write detailed, narrative hypotheses in clear marketing language, not just bullet points.

Your job:
- Read the media pattern for a given channel (spend, support, CPP changes vs last year/previous period).
- Interpret what is happening to its cost structure, delivery, and efficiency.
- Link this to likely movement in:
  * Effectiveness = Incremental Volume / Support
  * ROAS = Incremental Volume / Spend
- Generate 2–4 short paragraphs that sound like what a CMO, media director,
  or MMM lead would write in a QBR/Media Review deck.

Guidelines:
- Use precise marketing language: CPM/CPC, reach, frequency, auction pressure, audience saturation,
  creative wear-out, targeting changes, platform algorithm shifts, etc., where relevant.
- Explicitly comment on Effectiveness and ROAS implications using the text provided.
- Avoid generic statements like "might be due to various factors"; be concrete and contextual.
- Do NOT invent exact numeric values for effectiveness or ROAS; stay directional.
- Avoid bullet lists; write cohesive paragraphs.
        """,
    }

    user_msg = {
        "role": "user",
        "content": f"""
Business vertical: {business_vertical}

Media tactic (variable): {variable}

Raw metrics (previous period vs current period):
- {spend_prev_col}: {row[spend_prev_col]}
- {spend_curr_col}: {row[spend_curr_col]}
- Spend change vs previous period: {fmt_pct(spend_pct)}

- {support_prev_col}: {row[support_prev_col]}
- {support_curr_col}: {row[support_curr_col]}
- Support change vs previous period: {fmt_pct(support_pct)}

- {cpp_prev_col}: {row[cpp_prev_col]}
- {cpp_curr_col}: {row[cpp_curr_col]}
- CPP change vs previous period: {fmt_pct(cpp_pct)}

Qualitative interpretation of movements:
- {classify_change(spend_pct, 'Spend')}
- {classify_change(support_pct, 'Support (impressions/GRPs/clicks)')}
- {classify_change(cpp_pct, 'CPP (cost per impression/GRP/click)')}

Effectiveness and ROAS implications (directional, pre-computed):
- {eff_text}
- {roas_text}

Please write a detailed marketing-media hypothesis for this specific channel in the context of {business_vertical}:
- Explain what is happening to this channel's efficiency and cost structure.
- Link back explicitly to Effectiveness (incremental volume per unit of support)
  and ROAS (incremental volume per unit of spend), using the given directional assessment.
- Call out 2–3 plausible drivers (for example: auction pricing inflation, over-targeting narrow audiences,
  frequency overshoot, creative fatigue, platform policy changes, shifting mix to weaker inventory, etc.).
- Make it sound like a commentary slide in a MMM / media performance review presentation.
""",
    }

    return [system_msg, user_msg]


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Media Hypothesis Generation (Excel-driven)", page_icon="📊")

st.title("📊 Media Hypothesis Generation from MMM EDA")
st.write(
    "Select a business vertical and media tactic. The app will read previous vs current period "
    "Spend, Support, and CPP changes from the Excel file and generate a detailed "
    "marketing-media hypothesis, including implications for Effectiveness and ROAS."
)

# Load data
try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Error loading data file '{DATA_PATH}': {e}")
    st.stop()

# Basic sanity check on required columns
required_base_cols = [
    "Variable",
    "Spend Change",
    "Support Change",
    "CPP Change",
]
missing = [c for c in required_base_cols if c not in df.columns]
if missing:
    st.error(f"The following required columns are missing in {DATA_PATH}: {missing}")
    st.stop()

# Detect dynamic period columns for Spend, Support, CPP
try:
    spend_prev_col, spend_curr_col = detect_period_columns(df, "Spend ")
    support_prev_col, support_curr_col = detect_period_columns(df, "Support ")
    cpp_prev_col, cpp_curr_col = detect_period_columns(df, "CPP ")
except Exception as e:
    st.error(f"Error detecting period columns: {e}")
    st.stop()

# Business vertical selection
business_vertical = st.selectbox(
    "MMM client business vertical",
    ["Retail", "CPG", "Insurance", "Finance", "Gaming", "Pharma", "Other"],
    index=0,
)

# Media tactic selection
variables = df["Variable"].astype(str).unique().tolist()
variables.sort()
selected_var = st.selectbox("Select media tactic (Variable from EDA)", variables)

# Locate the row for the selected variable
row = df[df["Variable"].astype(str) == selected_var].iloc[0]

# Show raw metrics for transparency (wide format)
st.markdown("### Selected tactic metrics (previous vs current period)")
display_cols = [
    spend_prev_col,
    spend_curr_col,
    "Spend Change",
    support_prev_col,
    support_curr_col,
    "Support Change",
    cpp_prev_col,
    cpp_curr_col,
    "CPP Change",
]
metrics_df = row[display_cols].to_frame().T
st.dataframe(metrics_df)

# Compute clean % changes
spend_pct = parse_pct_value(row["Spend Change"])
support_pct = parse_pct_value(row["Support Change"])
cpp_pct = parse_pct_value(row["CPP Change"])

# Pre-compute directional Effectiveness & ROAS narrative
eff_text, roas_text = infer_effectiveness_and_roas(spend_pct, support_pct, cpp_pct)

if st.button("Generate detailed media hypothesis"):
    with st.spinner("Thinking like an MMM media strategist..."):
        try:
            messages = build_prompt(
                business_vertical,
                selected_var,
                row,
                spend_prev_col,
                spend_curr_col,
                support_prev_col,
                support_curr_col,
                cpp_prev_col,
                cpp_curr_col,
                spend_pct,
                support_pct,
                cpp_pct,
                eff_text,
                roas_text,
            )
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.4,
            )
            hypothesis = response.choices[0].message.content

            st.markdown("### Detailed Media Hypothesis")
            st.markdown(hypothesis)

            st.markdown("### Directional Effectiveness & ROAS Summary")
            st.markdown(f"- {eff_text}")
            st.markdown(f"- {roas_text}")

        except Exception as e:
            st.error(f"Error calling OpenAI API: {e}")
