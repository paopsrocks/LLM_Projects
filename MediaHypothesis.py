# %%
import os
from dotenv import load_dotenv
from openai import OpenAI

#import panel as pn
#pn.extension()
import streamlit as st

load_dotenv()   # loads .env into environment

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = "gpt-4o-mini"

# %%
def continue_conversation(messages, temperature = 0):
    response = client.chat.completions.create(
        model = model,
        messages = messages,
        temperature = temperature,
    )
    return response.choices[0].message.content

# %% [markdown]
# # Streamlit App Setup

# %%
st.set_page_config(page_title="Media Hypothesis Generation Chatbot", page_icon="📺")

st.title("📺 Media Hypothesis Generation Chatbot")
st.write(
    "Enter a media tactic or performance pattern, and I'll generate a structured MMM-style explanation and hypotheses."
)

# %%
# Initialize conversation state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": """
You are a Media Hypothesis Assistant for a Marketing Mix Modeling (MMM) team.

Your job:
- The user will typically type a media tactic name or a short pattern description.
- You should identify the tactic and return:
  1) A short, clear **definition**.
  2) Its **role in the consumer funnel** (e.g., awareness, consideration, conversion, loyalty).
  3) The typical **support metric** used (e.g., GRPs, impressions, clicks).
  4) The typical **business impact** (incremental volume, ROAS, brand KPIs).
  5) How this tactic usually appears in **MMM models** (e.g., adstocked GRPs, impressions, clicks).

Known media tactics in this project (examples):
- CTV
- Digital Circular
- Google Ads / Paid Search Branded / Paid Search UnBranded
- Online Display
- Online Video
- Paid Social
- Influencer
- Shopper Marketing
- Direct Mail
- Linear TV
- Public Relations
- App Downloads / App Installs campaigns

The MMM client can belong to different business verticals such as **Retail, CPG, Insurance, Finance, Gaming, Pharma, or Other**. Whenever this business context is provided, you must explicitly take it into account and adapt your definitions and hypotheses to that vertical.

When the user ONLY gives a tactic name:
- Focus on **Definition + Role in funnel + Typical support & pricing + MMM perspective**.

When the user ALSO mentions performance patterns like:
- Spend % change
- Support % change (impressions, clicks, GRPs)
- CPP % change (cost per point / cost per impression / cost per click)
- Effectiveness change (Incremental Volume / Support)
- ROAS change

Then, in addition to the definition, also:
- Propose **2–3 concise hypotheses** for what might be happening.
- Use MMM language: incremental volume, effectiveness, ROAS, CPP, support.
- Be diagnostic and cautious, not absolute (e.g., "One possible explanation is...").

Formatting:
- Use short sections with headings:
  - **Definition**
  - **Role in funnel**
  - **Typical support & pricing**
  - **MMM perspective**
  - **Hypotheses** (only if patterns are mentioned)

Tone:
- Professional, clear, concise.
- Assume the user is an MMM analyst.
            """
        }
    ]

# %%

# ---------- Business / vertical selection ----------
business_vertical = st.selectbox(
    "MMM client business vertical",
    ["Retail", "CPG", "Insurance", "Finance", "Gaming", "Pharma", "Other"],
    index=0,
)

# ---------- User input ----------
user_input = st.text_input(
    "Media tactic or pattern",
    value="Paid Search Branded",
    placeholder='e.g. "CTV", or "Paid Social – spend up 30%, impressions flat, ROAS down"',
)

# %%
if st.button("Generate hypothesis"):
    text = user_input.strip()
    if text:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": text})

        # Create a copy of messages with extra business context
        messages_for_api = st.session_state.messages + [
            {
                "role": "user",
                "content": f"The MMM client business vertical is: {business_vertical}. "
                           f"Please tailor your definitions and hypotheses to this {business_vertical} context.",
            }
        ]

        # Get assistant response
        try:
            reply = continue_conversation(messages_for_api)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            st.error(f"Error calling OpenAI API: {e}")


# ---------- Show conversation ----------
st.markdown("---")
st.subheader("Conversation")

for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    elif msg["role"] == "assistant":
        st.markdown(f"**Assistant:**\n\n{msg['content']}")


