from __future__ import annotations

import re
import subprocess
from pathlib import Path

import cv2
import whisper
from PIL import Image, ImageDraw
from youtube_transcript_api import YouTubeTranscriptApi

from .config import DEFAULT_CAPTURE_FPS, OUTPUT_DIR, WHISPER_MODEL_NAME


whisper_model = whisper.load_model(WHISPER_MODEL_NAME)


def extract_video_id(url: str) -> str | None:
    match = re.search(r"(?:v=|youtu\.be/|/live/|embed/)([A-Za-z0-9_-]{11})", url)
    return match.group(1) if match else None


def extract_frames(video_path: str | Path, n: int = 8) -> list[dict[str, Image.Image | float]]:
    capture = cv2.VideoCapture(str(video_path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or DEFAULT_CAPTURE_FPS
    frames: list[dict[str, Image.Image | float]] = []

    for index in [int(total * i / n) for i in range(n)]:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = capture.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append({"image": Image.fromarray(rgb), "ts": index / fps})

    capture.release()
    return frames


def make_frame_grid(frames: list[dict[str, Image.Image | float]], cols: int = 4) -> Image.Image:
    thumb_w, thumb_h, label_h = 240, 135, 24
    rows = (len(frames) + cols - 1) // cols
    grid = Image.new("RGB", (thumb_w * cols, (thumb_h + label_h) * rows), (255, 235, 245))
    draw = ImageDraw.Draw(grid)

    for index, frame in enumerate(frames):
        col, row = index % cols, index // cols
        x = col * thumb_w
        y = row * (thumb_h + label_h)
        image = frame["image"]
        timestamp = float(frame["ts"])
        grid.paste(image.resize((thumb_w, thumb_h), Image.LANCZOS), (x, y))
        label = f"  {int(timestamp // 60):02d}:{int(timestamp % 60):02d}"
        draw.rectangle([x, y + thumb_h, x + thumb_w, y + thumb_h + label_h], fill=(255, 210, 230))
        draw.text((x + 6, y + thumb_h + 5), label, fill=(136, 0, 68))

    return grid


def get_transcript(video_id: str) -> tuple[str | None, str | None]:
    try:
        entries = YouTubeTranscriptApi.get_transcript(video_id)
    except Exception:
        return None, None

    text = " ".join(entry["text"] for entry in entries)
    stamped = "\n".join(
        f"[{int(entry['start'] // 60):02d}:{int(entry['start'] % 60):02d}] {entry['text']}"
        for entry in entries
    )
    return text, stamped


def transcribe_audio(video_path: str | Path) -> tuple[str, str]:
    audio_path = OUTPUT_DIR / "audio.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(audio_path),
            "-loglevel",
            "error",
        ],
        check=True,
    )

    result = whisper_model.transcribe(str(audio_path), verbose=False)
    text = result["text"]
    stamped = "\n".join(
        f"[{int(segment['start'] // 60):02d}:{int(segment['start'] % 60):02d}] {segment['text']}"
        for segment in result.get("segments", [])
    )
    return text, stamped
