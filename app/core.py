from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path

import ollama_client as ollama

from guide_questions import GUIDE_QUESTIONS

from parser import (

    Segment,

    expert_only,

    extract_expert_label,

    parse_transcript,

)

from retrieval import Match, Retriever



STOPWORDS = {

    "the", "a", "an", "is", "are", "of", "to", "in", "for", "and", "on",

    "that", "this", "with", "as", "it", "be", "by", "or", "we", "i", "you",

    "your", "our", "their", "than", "but", "so", "it's",

}



@dataclass

class Transcript:

    label: str

    all_segments: list[Segment]

    segments: list[Segment]

    retriever: Retriever


# LOAD TRANSCRIPTS


def load_transcripts(

    directory: Path,

    files: dict[str, str] | None = None

) -> list[Transcript]:

    if files:

        source = list(files.items())

    else:

        source = [

            (

                p.stem,

                p.read_text(encoding="utf-8")

            )

            for p in sorted(directory.glob("*.txt"))

        ]

    transcripts = []

    for default_label, raw_text in source:

        label = extract_expert_label(

            raw_text,

            default_label

        )

        full_segments = parse_transcript(

            raw_text,

            expert_label=label

        )

        expert_segments = expert_only(

            full_segments

        )

        transcripts.append(

            Transcript(

                label=label,

                all_segments=full_segments,

                segments=expert_segments,

                retriever=Retriever(expert_segments),

            )

        )

    return transcripts


# HELPERS


def _short_expert_name(

    label: str

) -> tuple[str, str | None]:

    market = None

    name = label

    if " - " in label:

        market, name = label.split(" - ", 1)

    if "(" in name:

        name = name[:name.index("(")].strip()

    return name, market



def _first_sentence(

    text: str,

    max_chars: int = 180

) -> str:

    text = text.strip()

    if not text:

        return ""

    for end in [".", "!", "?"]:

        idx = text.find(end)

        if 0 < idx < max_chars:

            return text[:idx + 1].strip()

    if len(text) > max_chars:

        return text[:max_chars].rsplit(" ", 1)[0] + "…"

    return text



# GUIDE QUESTIONS


def answer_guide_questions(

    transcripts: list[Transcript],

    top_k: int = 1

) -> dict:

    results = {}

    for question in GUIDE_QUESTIONS:

        key = question["key"]

        question_text = question["question"]

        results[key] = {}

        for transcript in transcripts:

            matches = transcript.retriever.search(

                question_text,

                top_k=top_k

            )

            results[key][transcript.label] = matches

    return results



# QUESTION SUMMARY


def synthesize_question_summary(

    question_topic: str,

    matches_by_expert: dict[str, list[Match]],

    use_llm: bool

) -> str:

    context_lines = []

    for expert, matches in matches_by_expert.items():

        if not matches:

            continue

        short_name, market = _short_expert_name(expert)

        location = f" ({market})" if market else ""

        context_lines.append(

            f'- {short_name}{location}: "{matches[0].segment.text}"'

        )

    if not context_lines:

        return "No insights retrieved."

    context = "\n".join(context_lines)


    # LLM SUMMARY

    if use_llm:

        prompt = (

            "Summarize the expert interview excerpts below for the "

            "given market research topic.\n\n"

            f"Topic: {question_topic}\n\n"

            "Write one natural paragraph of 3-4 sentences. "

            "Start with the main takeaway, then explain what the "

            "experts said. Mention the expert name and country when "

            "sharing their view. Show agreement or differences when "

            "they are clear from the excerpts. "

            "Use only the information in the excerpts and do not "

            "add anything from outside. "

            "Do not mention transcript numbers or use bullet points. "

            "Return only the summary paragraph.\n\n"

            f"Expert excerpts:\n{context}\n\n"

            "Summary:"

        )

        response = ollama.generate(prompt)

        if response:

            return response.strip()

    # FALLBACK


    openings = {

        "Current adoption":

            "Across the three markets, robotic surgery adoption is growing, although the pace and level of adoption vary by hospital type.",

        "Main barriers to adoption":

            "Cost, hospital budgets, and limited resources emerge as some of the main barriers to wider adoption.",

        "Budget & ROI importance":

            "Hospital budgets and expected ROI appear to play an important role in purchasing decisions across the three markets.",

        "Training & clinical outcomes":

            "Surgeon training and clinical outcomes are consistently highlighted as important considerations for robotic surgery adoption.",

        "3-4 year adoption outlook":

            "The experts generally expect robotic surgery adoption to continue increasing over the next three to four years, although the pace may vary across hospitals and markets.",

        "Purchasing decision timeline":

            "The purchasing process for a new robotic surgery system can take considerable time and usually involves several factors before a decision is made.",

    }

    opening = openings.get(

        question_topic,

        "The experts highlight several factors that shape robotic surgery adoption and purchasing decisions."

    )

    expert_text = []

    for expert, matches in matches_by_expert.items():

        if not matches:

            continue

        name, market = _short_expert_name(expert)

        text = matches[0].segment.text.strip()

        if not text:

            continue

        expert_label = name

        if market:

            expert_label += f" ({market})"

        expert_text.append(

            f"According to {expert_label}, "

            f"{text[0].lower() + text[1:]}"

        )

    if expert_text:

        return opening + " " + " ".join(expert_text)

    return opening


# FREE QUESTION

def answer_free_question(

    question: str,

    transcripts: list[Transcript],

    use_llm: bool

) -> dict:

    matches_by_expert = {}

    for transcript in transcripts:

        matches = transcript.retriever.search(

            question,

            top_k=2

        )

        matches_by_expert[transcript.label] = matches

    all_matches = []

    for matches in matches_by_expert.values():

        all_matches.extend(matches)

    all_matches.sort(

        key=lambda match: match.score,

        reverse=True

    )

    citations = all_matches[:6]

    if not citations:

        return {

            "answer": "I couldn't find a relevant answer in the transcripts.",

            "citations": [],

            "mode": "no_match",

        }

    # FALLBACK WITHOUT LLM

    if not use_llm:

        return {

            "answer": None,

            "citations": citations,

            "mode": "retrieval",

        }


    # BUILD LLM CONTEXT


    context_lines = []

    for transcript in transcripts:

        expert = transcript.label

        matches = matches_by_expert.get(

            expert,

            []

        )

        for match in matches:

            name, market = _short_expert_name(expert)

            expert_label = name

            if market:

                expert_label += f" ({market})"

            context_lines.append(

                f'- {expert_label} | '

                f'{match.segment.timestamp}: '

                f'"{match.segment.text}"'

            )

    context = "\n".join(context_lines)

    prompt = (

        "Answer the user's question using the expert excerpts below.\n\n"

        f"Question: {question}\n\n"

        "Write one natural paragraph of 3-4 sentences. "

        "Start with the main answer, then include the relevant "

        "expert views with their names and countries. "

        "Only use information from the excerpts. "

        "Do not add information or assumptions that are not supported "

        "by the excerpts. Keep the wording close to what the experts "

        "said. Do not mention transcript numbers. "

        "Return only the answer paragraph.\n\n"

        f"Transcript excerpts:\n{context}\n\n"

        "Answer:"

    )

    response = ollama.generate(prompt)

    if response:

     answer_text = response.strip()

    summary_prompt = (

        "Summarize the expert answer below in simple "

        "market-research language.\n\n"

        "Write 2-3 clear sentences with the main takeaway. "

        "Do not repeat the question or mention transcript numbers. "

        "Use only information from the answer and do not add anything.\n\n"

        f"Answer:\n{answer_text}\n\n"

        "Summary:"

    )

    summary_response = ollama.generate(summary_prompt)

    return {

        "answer": answer_text,

        "summary": (

            summary_response.strip()

            if summary_response

            else answer_text

        ),

        "citations": citations,

        "mode": "llm_verified",

    }

    return {

        "answer": None,

        "citations": citations,

        "mode": "retrieval",

    }


# COMPARE ACROSS EXPERTS


def compare_across_experts(

    guide_answers: dict,

    use_llm: bool

) -> dict:

    comparisons = {}

    for question in GUIDE_QUESTIONS:

        key = question["key"]

        topic = question["topic"]

        matches_by_expert = guide_answers.get(

            key,

            {}

        )

        available = {

            expert: matches

            for expert, matches in matches_by_expert.items()

            if matches

        }

        synthesis = synthesize_question_summary(

            topic,

            matches_by_expert,

            use_llm

        )

        # Check if there is enough evidence to compare

        # the expert responses.

        if len(available) < 2:

            label = "agree"

        else:

            texts = [

                matches[0].segment.text.lower()

                for matches in available.values()

                if matches

            ]

            word_sets = []

            for text in texts:

                words = {

                    word.strip(".,;:!?()[]\"'")

                    for word in text.split()

                    if len(word.strip(".,;:!?()[]\"'")) > 3

                    and word.lower() not in STOPWORDS

                }

                word_sets.append(words)

            shared_words = set()

            if word_sets:

                shared_words = set.intersection(

                    *word_sets

                )

            # Simple check based on words shared between excerpts.

            # More shared terms usually means the experts are

            # discussing a similar point.

            if len(shared_words) >= 2:

                label = "agree"

            else:

                label = "disagree"

        comparisons[key] = {

            "label": label,

            "synthesis": synthesis,

            "matches": matches_by_expert,

        }

    return comparisons
