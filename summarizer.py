from __future__ import annotations

import json
import re
from collections import Counter

from groq import Groq

from .config import GROQ_API_KEY, GROQ_MODEL_NAME


_stopwords = {
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "of", "and", "or",
    "but", "for", "with", "this", "that", "was", "are", "be", "as", "by",
    "we", "he", "she", "they", "you", "i", "has", "have", "had", "will",
    "its", "from", "about", "been", "were", "so", "if", "not", "what",
}


def _local_summary(transcript: str, stamped_transcript: str = "") -> dict[str, list[str] | str]:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", transcript) if len(sentence.strip()) > 40]
    words = re.findall(r"\b\w+\b", transcript.lower())
    frequency = Counter(word for word in words if word not in _stopwords and len(word) > 3)

    def score(sentence: str) -> int:
        return sum(frequency.get(word, 0) for word in re.findall(r"\b\w+\b", sentence.lower()))

    ranked = sorted(enumerate(sentences), key=lambda item: score(item[1]), reverse=True)
    top_indices = sorted(index for index, _ in ranked[:8])
    key_sentences = [sentences[index] for index in top_indices]

    if stamped_transcript:
        lines = stamped_transcript.strip().split("\n")
        step = max(1, len(lines) // 6)
        key_points = lines[::step][:6]
    else:
        key_points = [f"• {sentence[:100]}" for sentence in key_sentences[:5]]

    topics = [word for word, _ in frequency.most_common(8) if len(word) > 4]
    third = max(1, len(key_sentences) // 3)
    summary = "\n\n".join(
        [
            " ".join(key_sentences[:third]),
            " ".join(key_sentences[third:2 * third]),
            " ".join(key_sentences[2 * third:]),
        ]
    ).strip()
    if not summary:
        summary = transcript[:500]

    return {"summary": summary, "key_points": key_points, "topics": topics, "events": []}


def ask_groq(prompt: str, max_tokens: int = 1000) -> str:
    if not GROQ_API_KEY:
        return ""

    client = Groq(api_key=GROQ_API_KEY)
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return ""


def generate_summary(transcript: str, title: str, visual_notes: str = "", stamped_transcript: str = "") -> dict[str, list[str] | str]:
    prompt = f"""You are analyzing a video titled: "{title}"

TRANSCRIPT:
{transcript[:5000]}

VISUAL FRAME ANALYSIS:
{visual_notes}

Return ONLY valid JSON, no markdown fences, no extra text:
{{
  "summary": "<3 paragraph overview of the entire video>",
  "key_points": ["[MM:SS] key point 1", "[MM:SS] key point 2", "[MM:SS] key point 3"],
  "topics": ["topic1", "topic2", "topic3"],
  "events": ["event1", "event2"]
}}"""

    raw = ask_groq(prompt, max_tokens=1000)
    if not raw:
        return _local_summary(transcript, stamped_transcript)

    try:
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(cleaned)
    except Exception:
        return {"summary": raw[:1000], "key_points": [], "topics": [], "events": []}


def ask_yt(question: str, context: str, title: str) -> str:
    if not question.strip():
        return "Please type a question."
    if not context:
        return "Analyze a video first, then ask questions."

    prompt = f"""You are answering questions about a video titled: "{title}"

Use ONLY the transcript below to answer. Be concise. Cite timestamps when possible.

TRANSCRIPT:
{context[:6000]}

QUESTION: {question}"""

    answer = ask_groq(prompt, max_tokens=500)
    return answer if answer else "Could not generate an answer. Please try again."


def format_analysis(data: dict[str, list[str] | str], title: str, duration: str = "") -> tuple[str, str, str]:
    summary_md = f"## {title}\n{duration}\n\n{data.get('summary', '')}"

    key_points = data.get("key_points", [])
    key_points_md = "## Key Points\n" + ("\n".join(f"- {item}" for item in key_points) if key_points else "None extracted.")

    topics = data.get("topics", [])
    events = data.get("events", [])
    topics_events_md = (
        "## Topics\n"
        + (", ".join(f"`{topic}`" for topic in topics) if topics else "N/A")
        + "\n\n## Events\n"
        + ("\n".join(f"- {event}" for event in events) if events else "None detected.")
    )
    return summary_md, key_points_md, topics_events_md
