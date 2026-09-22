from pathlib import Path

import streamlit as st

import ollama_client as ollama

from core import (

    answer_free_question,

    answer_guide_questions,

    compare_across_experts,

    load_transcripts,

    synthesize_question_summary,

)

from guide_questions import GUIDE_QUESTIONS



BASE_DIR = Path(__file__).resolve().parent

TRANSCRIPTS_DIR = BASE_DIR.parent / "transcripts"

# color for each expert

EXPERT_COLORS = ["#2a78d6", "#c9622a", "#1a9c6e"]



st.set_page_config(page_title="Robotic Surgery Insights", layout="wide")



# sidebar styles

st.markdown(

    """

    <style>

    /* sidebar */

    [data-testid="stSidebar"] {

        background: linear-gradient(180deg, #13172e 0%, #0b0d18 100%) !important;

        padding-top: 1rem !important;

    }

    [data-testid="stSidebar"] *, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {

        color: #e2e8f0 !important;

    }

    /* sidebar title */

    .sidebar-brand-title {

        font-size: 18px;

        font-weight: 800;

        letter-spacing: 1.5px;

        color: #00d2ff !important;

        padding: 10px 0 20px 0;

        border-bottom: 1px solid rgba(255,255,255,0.08);

        margin-bottom: 25px;

        display: flex;

        align-items: center;

        gap: 8px;

    }

    .sidebar-brand-title span {

        color: #ffffff !important;

        font-weight: 400;

    }

    /* selected nav item */

    [data-testid="stSidebar"] [aria-checked="true"] ~ div p {

        color: #13172e !important;

        font-weight: 700 !important;

    }

    [data-testid="stSidebar"] [aria-checked="true"] ~ div {

        background-color: #ffffff !important;

        border-radius: 8px !important;

        padding: 6px 12px !important;

        box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;

    }

    /* expander */

    [data-testid="stSidebar"] .stExpander {

        background-color: rgba(255,255,255,0.04) !important;

        border-radius: 8px !important;

        border: 1px solid rgba(255,255,255,0.08) !important;

    }

    </style>

    """,

    unsafe_allow_html=True,

)



@st.cache_resource(show_spinner="Loading transcripts...")

def get_transcripts(uploaded_signature):

    if uploaded_signature:

        return load_transcripts(TRANSCRIPTS_DIR, files=dict(uploaded_signature))

    return load_transcripts(TRANSCRIPTS_DIR)



@st.cache_resource(show_spinner=False)

def get_ollama_status():

    return ollama.is_available()



def short_name(label: str) -> str:

    return label



def colored_name(label: str, color: str) -> str:

    return f'<span style="color:{color}; font-weight:600">{short_name(label)}</span>'



def render_quote(

    label: str,

    color: str,

    text: str,

    timestamp: str,

    score: float | None = None,

):

    with st.container(border=True):

        st.markdown(colored_name(label, color), unsafe_allow_html=True)

        st.markdown(f"> {text}")

        st.caption(f"**Timestamp**: `{timestamp}`")



with st.sidebar:

    st.markdown(

        '<div class="sidebar-brand-title">ROBOTIC <b>SURGERY</b> INSIGHTS</div>',

        unsafe_allow_html=True,

    )

    nav_option = st.radio(

        "NAVIGATION",

        options=[

            "Home / Ask a Question",

            "Interview Guide Answers",

            "Themes & Disagreements",

        ],

        index=0,

        label_visibility="collapsed",

    )

    llm_available = get_ollama_status()

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("Upload Transcripts")

    uploaded = st.file_uploader(

        "Upload custom transcripts (.txt)",

        type="txt",

        accept_multiple_files=True,

    )

    uploaded_signature = None

    if uploaded:

        uploaded_signature = tuple(

            (

                f.name.rsplit(".", 1)[0],

                f.getvalue().decode("utf-8")

            )

            for f in uploaded

        )

        st.caption(f"Loaded {len(uploaded)} custom transcript(s).")



transcripts = get_transcripts(uploaded_signature)

if not transcripts:

    st.error(

        f"No transcripts found in {TRANSCRIPTS_DIR}. "

        "Add .txt files or upload some in the sidebar."

    )

    st.stop()



guide_answers = answer_guide_questions(transcripts, top_k=1)

expert_labels = [t.label for t in transcripts]

colors = {

    label: EXPERT_COLORS[i % len(EXPERT_COLORS)]

    for i, label in enumerate(expert_labels)

}

# PAGE 1: HOME / ASK A QUESTION

if nav_option == "Home / Ask a Question":

    st.title("Robotic Surgery Adoption in Europe")

    st.markdown(

        "#### **Expert Insights** · Market Research (France, Germany, UK)"

    )

    st.caption("Explore insights from the expert transcripts.")

    st.markdown("---")

    st.header("Ask a Question Across All Transcripts")

    query = st.text_input(

        "Type your query and press Enter:",

        placeholder="e.g. Why is the UK slower than Germany to adopt robotic systems?",

        key="main_page_query",

    )

    if query.strip():

     st.session_state["last_query"] = query

     with st.spinner("Searching expert transcripts..."):

            st.session_state["last_result"] = answer_free_question(

                query,

                transcripts,

                use_llm=llm_available

            )

    result = st.session_state.get("last_result") if query.strip() else None

    if result:

        st.markdown("#### Search Result")

        if result["mode"] == "llm_verified":

            st.markdown("##### Summary")

            st.write(result["summary"])

            st.caption("Summary based on the expert transcripts.")

        elif result["mode"] == "no_match":

            st.error(result["answer"])

        if result["citations"]:

            st.markdown("##### Source Excerpts")

            for m in result["citations"]:

                color = colors.get(m.segment.expert, "#666666")

                render_quote(

                    m.segment.expert,

                    color,

                    m.segment.text,

                    m.segment.timestamp,

                    m.score

                )

# PAGE 2: INTERVIEW GUIDE ANSWERS

elif nav_option == "Interview Guide Answers":

    st.title("Interview Guide Answers")

    st.caption("Summaries and transcript evidence for each question.")

    for q in GUIDE_QUESTIONS:

        with st.expander(

            f"{q['topic']} — {q['question']}",

            expanded=False

        ):

            summary = synthesize_question_summary(

    q["topic"],

    guide_answers[q["key"]],

    llm_available

)

            st.info(f"**Summary**  \n\n{summary}")

            st.markdown("#### Transcript Evidence")

            for label in expert_labels:

                matches = guide_answers[q["key"]][label]

                if not matches:

                    st.markdown(

                        colored_name(label, colors[label]),

                        unsafe_allow_html=True

                    )

                    st.caption("No relevant content found.")

                    continue

                m = matches[0]

                render_quote(

                    label,

                    colors[label],

                    m.segment.text,

                    m.segment.timestamp,

                    m.score

                )


# PAGE 3: THEMES & DISAGREEMENTS

elif nav_option == "Themes & Disagreements":

    st.title("Themes & Market Disagreements")

    st.caption("See where the experts agree and where their views differ.")

    comparisons = compare_across_experts(

        guide_answers,

        use_llm=llm_available

    )

    themes = [

        q

        for q in GUIDE_QUESTIONS

        if comparisons[q["key"]]["label"] == "agree"

    ]

    disagreements = [

        q

        for q in GUIDE_QUESTIONS

        if comparisons[q["key"]]["label"] == "disagree"

    ]

    st.markdown("### Common Themes")

    if not themes:

        st.caption("No strong common themes detected.")

    for q in themes:

        st.markdown(

            f"- **{q['topic']}**: "

            f"{comparisons[q['key']]['synthesis']}"

        )

    st.divider()

    st.markdown("### Key Market Disagreements")

    if not disagreements:

        st.caption("No notable disagreements detected.")

    for i, q in enumerate(disagreements, start=1):

        comp = comparisons[q["key"]]

        st.markdown(f"#### {i}. {q['topic']}")

        st.info(comp["synthesis"])

        for t in transcripts:

            matches = comp["matches"].get(t.label, [])

            if matches:

                render_quote(

                    t.label,

                    colors[t.label],

                    matches[0].segment.text,

                    matches[0].segment.timestamp

                )

            else:

                st.markdown(

                    colored_name(t.label, colors[t.label]),

                    unsafe_allow_html=True

                )

        st.divider()

