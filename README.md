# Robotic Surgery Adoption in Europe — Expert Call Analyzer

I built a small local app for a case study to analyze 3 expert-call transcripts
from the same market research project on robotic surgery adoption in France,
Germany, and the UK.

# What it does

- Reads the 3 transcripts that come with the project, or lets you upload your own
  transcripts in the same format.
- Answers each interview guide question for each expert using the actual quote
  and timestamp from the transcript.
- Compares the three experts for each question to see where their views are
  similar and where they differ.
- Includes a free-text box to ask your own question across all three transcripts.

The answers are tied back to the transcript quotes, so the source text is always
available for checking.

# How it's put together

The main goal was to avoid relying on an LLM to generate answers that may sound
right but are not actually in the transcripts.

The app has two main parts:

1. Retrieval

   `retrieval.py` uses TF-IDF and cosine similarity to find the transcript
   sections that are most relevant to a question.

   It doesn't need a model download or API, so the retrieval part is
   deterministic. This is used for the "Interview Guide Answers" and
   "Themes" tabs.

2. LLM (optional)

   The LLM is only used after the relevant transcript sections have already
   been retrieved. It is used for the comparison between experts and for
   answering free-text questions.

   It runs locally through Ollama, so no API key is required.

   For the Q&A tab, the LLM also returns the expert and timestamp for its
   citations. The app checks these against the retrieved transcript sections.
   If the citations don't match, or if Ollama isn't running, it falls back to
   showing the original quotes.

3. Works without the LLM

   Ollama is optional. If it isn't available, the app still works using the
   retrieved quotes and timestamps instead of generated answers.

# Project layout

robotic-surgery-insights/

├── transcripts/             3 sample expert-call transcripts (timestamped .txt)
├── app/
│   ├── app.py               Streamlit UI
│   ├── core.py              guide answers, cross-expert comparison, Q&A logic
│   ├── retrieval.py         TF-IDF search
│   ├── parser.py            turns raw transcript text into segments
│   ├── guide_questions.py   the 6 interview guide questions
│   └── ollama_client.py     local Ollama connection, if available
├── requirements.txt
└── README.md

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Aditee1511/robotic-surgery-insights.git
cd robotic-surgery-insights
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

### 3. Activate the virtual environment

**macOS / Linux:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the application

```bash
streamlit run app/app.py
```

The application will open in your browser at:

`http://localhost:8501`

### Notes

* The application runs locally using the transcripts provided in the `transcripts/` folder.
* No external database or cloud deployment is required to run the application.
* The optional Ollama integration is included in the project for local LLM-based functionality.
