from __future__ import annotations

import cv2
import subprocess

from .config import OUTPUT_DIR
from .dip import make_dip_grid, run_dip
from .media import extract_frames, extract_video_id, get_transcript, transcribe_audio, make_frame_grid
from .summarizer import ask_yt, format_analysis, generate_summary


current_transcript = {"text": "", "title": ""}
stamped_global = ""


def _duration_seconds(video_path: str) -> int:
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    capture.release()
    return int(frame_count / fps) if fps else 0


def run_youtube_pipeline(url: str, status_fn=print):
    global stamped_global
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    clean_url = f"https://www.youtube.com/watch?v={video_id}"

    title_result = subprocess.run(
        ["yt-dlp", "--get-title", "--no-playlist", clean_url],
        capture_output=True,
        text=True,
        timeout=30,
    )
    title = title_result.stdout.strip().split("\n")[0] or "YouTube Video"
    status_fn(f"Found: {title}")

    status_fn("Downloading video...")
    out_path = str(OUTPUT_DIR / "yt_video.mp4")

    clients = [
        ["--extractor-args", "youtube:player_client=android"],
        ["--extractor-args", "youtube:player_client=ios"],
        ["--extractor-args", "youtube:player_client=android,web"],
        ["--extractor-args", "youtube:player_client=tv_embedded"],
    ]

    downloaded = False
    for client_args in clients:
        command = ["yt-dlp", "--no-playlist", "--no-warnings", "-o", out_path, *client_args, clean_url]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            downloaded = True
            status_fn("Downloaded")
            break
        status_fn("Trying next client...")

    if not downloaded:
        raise ValueError("Download failed. Try a different video or a non-news channel.")

    status_fn("Getting transcript...")
    transcript, stamped = get_transcript(video_id)
    if transcript:
        status_fn("Official captions found")
    else:
        status_fn("No captions - running Whisper...")
        transcript, stamped = transcribe_audio(out_path)
        status_fn("Whisper transcription done")

    stamped_global = stamped or ""

    status_fn("Extracting keyframes...")
    converted_path = str(OUTPUT_DIR / "yt_converted.mp4")
    frames = extract_frames(out_path, n=8)
    if not frames:
        status_fn("Converting video format...")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                out_path,
                "-vf",
                "scale=640:-2",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-c:a",
                "aac",
                converted_path,
                "-loglevel",
                "error",
            ],
            check=True,
        )
        frames = extract_frames(converted_path, n=8)
    else:
        converted_path = out_path

    if not frames:
        raise ValueError("Could not extract frames.")

    frame_grid = make_frame_grid(frames)
    status_fn(f"{len(frames)} frames extracted")

    status_fn("Running DIP pipeline...")
    mid = frames[len(frames) // 2]["image"]
    dip_grid = make_dip_grid(mid)
    status_fn("DIP complete")

    status_fn("Analyzing frames with DIP...")
    visual_notes = ""
    for frame in frames[:8]:
        timestamp = float(frame["ts"])
        dip = run_dip(frame["image"])
        edge_density = float(dip["_edge_density"])
        focus_score = float(dip["_focus_score"])

        if edge_density > 0.15:
            content = "high activity / complex scene"
        elif edge_density > 0.08:
            content = "moderate activity / mixed content"
        else:
            content = "low activity / simple scene"

        if focus_score < 50:
            focus = "blurry/motion"
        elif focus_score < 200:
            focus = "moderate focus"
        else:
            focus = "sharp/clear"

        visual_notes += (
            f"[{int(timestamp // 60):02d}:{int(timestamp % 60):02d}] "
            f"Edge density: {edge_density:.3f} ({content}), "
            f"Focus: {focus_score:.1f} ({focus})\n"
        )

    status_fn("Generating summary...")
    duration_label = f"Duration: {_duration_seconds(converted_path) // 60}m {_duration_seconds(converted_path) % 60}s"
    data = generate_summary(transcript or "", title, visual_notes, stamped or "")
    summary_md, key_points_md, topics_events_md = format_analysis(data, title, duration_label)
    status_fn("Done")

    for match in OUTPUT_DIR.glob("yt_video.*"):
        try:
            match.unlink()
        except Exception:
            pass

    return summary_md, key_points_md, topics_events_md, frame_grid, dip_grid, stamped or "", transcript or ""


def run_live_pipeline(url: str, duration: int, status_fn=print):
    global stamped_global
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL - could not find video ID")

    clean_url = f"https://www.youtube.com/watch?v={video_id}"
    status_fn(f"Capturing {duration}s from live stream...")
    out_path = str(OUTPUT_DIR / "live_clip.mp4")

    title_result = subprocess.run(
        ["yt-dlp", "--get-title", "--no-playlist", clean_url],
        capture_output=True,
        text=True,
        timeout=30,
    )
    title = title_result.stdout.strip().split("\n")[0] or "Live Stream"

    stream_result = subprocess.run(
        ["yt-dlp", "-g", "-f", "best[height<=480]/best", "--no-playlist", clean_url],
        capture_output=True,
        text=True,
        timeout=30,
    )
    stream_urls = [url.strip() for url in stream_result.stdout.strip().split("\n") if url.strip()]
    if not stream_urls:
        raise ValueError("Could not get stream URL. Is the stream currently active?")

    status_fn(f"Recording {duration} seconds...")
    subprocess.run(
        ["ffmpeg", "-y", "-t", str(duration), "-i", stream_urls[0], "-c:v", "libx264", "-c:a", "aac", out_path, "-loglevel", "error"],
        check=True,
        timeout=int(duration) + 120,
    )
    status_fn("Clip captured")

    status_fn("Extracting frames...")
    frames = extract_frames(out_path, n=8)
    if not frames:
        raise ValueError("Could not extract frames from live clip.")
    frame_grid = make_frame_grid(frames)
    status_fn(f"{len(frames)} frames extracted")

    status_fn("Running DIP pipeline...")
    mid = frames[len(frames) // 2]["image"]
    dip_grid = make_dip_grid(mid)
    status_fn("DIP complete")

    status_fn("Transcribing with Whisper...")
    transcript, stamped = transcribe_audio(out_path)
    stamped_global = stamped
    status_fn("Transcription done")

    status_fn("Analyzing frames with DIP...")
    visual_notes = ""
    for frame in frames[:8]:
        timestamp = float(frame["ts"])
        dip = run_dip(frame["image"])
        edge_density = float(dip["_edge_density"])
        focus_score = float(dip["_focus_score"])

        if edge_density > 0.15:
            content = "high activity / complex scene"
        elif edge_density > 0.08:
            content = "moderate activity / mixed content"
        else:
            content = "low activity / simple scene"

        if focus_score < 50:
            focus = "blurry/motion"
        elif focus_score < 200:
            focus = "moderate focus"
        else:
            focus = "sharp/clear"

        visual_notes += (
            f"[{int(timestamp // 60):02d}:{int(timestamp % 60):02d}] "
            f"Edge density: {edge_density:.3f} ({content}), "
            f"Focus: {focus_score:.1f} ({focus})\n"
        )
    status_fn("Visual analysis done")

    status_fn("Generating summary...")
    data = generate_summary(transcript, title, visual_notes, stamped)
    summary_md, key_points_md, topics_events_md = format_analysis(data, f"LIVE: {title}")
    status_fn("Analysis complete")
    return summary_md, key_points_md, topics_events_md, frame_grid, dip_grid, stamped, transcript


def ask_question(question: str):
    return ask_yt(question, current_transcript["text"], current_transcript["title"])


def yt_analyze(url: str):
    log_lines: list[str] = []

    def log(message: str):
        log_lines.append(message)
        print(message)

    try:
        summary, key_points, topics_events, frames, dip, stamped, transcript = run_youtube_pipeline(url, status_fn=log)
        current_transcript["text"] = transcript
        current_transcript["title"] = url
        return summary, key_points, topics_events, frames, dip, stamped, "\n".join(log_lines)
    except Exception as exc:
        log(f"Error: {exc}")
        return f"Error: {exc}", "", "", None, None, "", "\n".join(log_lines)


def live_analyze(url: str, duration: int):
    log_lines: list[str] = []

    def log(message: str):
        log_lines.append(message)
        print(message)

    try:
        summary, key_points, topics_events, frames, dip, stamped, transcript = run_live_pipeline(url, int(duration), status_fn=log)
        current_transcript["text"] = transcript
        current_transcript["title"] = url
        return summary, key_points, topics_events, frames, dip, stamped, "\n".join(log_lines)
    except Exception as exc:
        log(f"Error: {exc}")
        return f"Error: {exc}", "", "", None, None, "", "\n".join(log_lines)
