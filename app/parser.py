from __future__ import annotations

import re

from dataclasses import dataclass

TIMESTAMP_REGEX = re.compile(r"^\[?(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\]?$")

SINGLE_LINE_REGEX = re.compile(
    r"^\[?(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\]?\s*(?P<speaker>[^:]+):\s*(?P<text>.+)$"
)

SPEAKER_TEXT_REGEX = re.compile(r"^(?P<speaker>[^:]+):\s*(?P<text>.+)$")


@dataclass
class Segment:

    expert: str    # transcript label, e.g. "France - Dr. Jean Martin (Head of Urology)"

    speaker: str   # speaker on this line, e.g. "Dr. Jean Martin" or "Interviewer"

    timestamp: str # "00:03:12"

    seconds: int

    text: str


def _to_seconds(ts: str) -> int:

    parts = [int(p) for p in ts.split(":")]

    if len(parts) == 3:

        return parts[0] * 3600 + parts[1] * 60 + parts[2]

    elif len(parts) == 2:

        return parts[0] * 60 + parts[1]

    return 0


def _format_ts(ts: str) -> str:

    parts = ts.split(":")

    if len(parts) == 2:

        return f"00:{parts[0].zfill(2)}:{parts[1].zfill(2)}"

    elif len(parts) == 3:

        return f"{parts[0].zfill(2)}:{parts[1].zfill(2)}:{parts[2].zfill(2)}"

    return ts


def parse_transcript(raw_text: str, expert_label: str) -> list[Segment]:

    segments: list[Segment] = []

    current_ts = None

    for line in raw_text.splitlines():

        line = line.strip()

        if not line:

            continue

        # Check single-line format: [00:00:18] Dr. Martin: text or 00:18 Dr. Martin: text

        single_match = SINGLE_LINE_REGEX.match(line)

        if single_match:

            raw_ts = single_match.group("ts")

            speaker = single_match.group("speaker").strip()

            text = single_match.group("text").strip()

            segments.append(

                Segment(

                    expert=expert_label,

                    speaker=speaker,

                    timestamp=_format_ts(raw_ts),

                    seconds=_to_seconds(raw_ts),

                    text=text,

                )

            )

            current_ts = None

            continue

        # Check standalone timestamp line: "00:18" or "[00:18]"

        ts_match = TIMESTAMP_REGEX.match(line)

        if ts_match:

            current_ts = ts_match.group("ts")

            continue

        # Check speaker line following a timestamp

        if current_ts:

            speaker_match = SPEAKER_TEXT_REGEX.match(line)

            if speaker_match:

                speaker = speaker_match.group("speaker").strip()

                text = speaker_match.group("text").strip()

                segments.append(

                    Segment(

                        expert=expert_label,

                        speaker=speaker,

                        timestamp=_format_ts(current_ts),

                        seconds=_to_seconds(current_ts),

                        text=text,

                    )

                )

                current_ts = None

    return segments


def expert_only(segments: list[Segment]) -> list[Segment]:

    # keep the interviewer questions out of retrieval

    return [s for s in segments if s.speaker.strip().lower() != "interviewer"]


def extract_expert_label(raw_text: str, default_label: str) -> str:
    """Extract a human-readable expert label from the transcript header.

    Expected header format (first few lines):
        Expert 1 - Dr. Jean Martin
        Role: Head of Urology
        Market: France

    Returns something like: "France - Dr. Jean Martin (Head of Urology)"
    Falls back to *default_label* when the header cannot be parsed.
    """
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()][:5]

    name = None
    role = None
    market = None

    for line in lines:
        # "Expert 1 - Dr. Jean Martin"  →  name = "Dr. Jean Martin"
        if not name and re.match(r"^Expert\s+\d+", line, re.IGNORECASE):
            parts = line.split("-", 1)
            if len(parts) == 2:
                name = parts[1].strip()
            continue

        # "Role: Head of Urology"
        if not role and line.lower().startswith("role:"):
            role = line.split(":", 1)[1].strip()
            continue

        # "Market: France"
        if not market and line.lower().startswith("market:"):
            market = line.split(":", 1)[1].strip()
            continue

    if name:
        label = f"{market} - {name}" if market else name
        if role:
            label += f" ({role})"
        return label

    return default_label