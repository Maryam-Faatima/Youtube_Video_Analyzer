from __future__ import annotations

import gradio as gr

from .dip import make_dip_grid
from .pipeline import ask_question, live_analyze, yt_analyze


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, body, .gradio-container {
    background-color: #000000 !important;
    color: #ffffff !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

.gradio-container {
    max-width: 1300px !important;
    margin: 0 auto !important;
    padding: 0 24px !important;
}

.gr-markdown h1 {
    font-size: 2.8rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #ff2d78, #ff85b3, #ff2d78) !important;
    background-size: 200% !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    animation: shimmer 3s infinite linear !important;
    margin-bottom: 4px !important;
}

@keyframes shimmer {
    0% { background-position: 0% }
    100% { background-position: 200% }
}

.gr-markdown h3 {
    color: #666666 !important;
    font-weight: 400 !important;
    font-size: 1rem !important;
    margin-top: 0 !important;
}

.tab-nav {
    background: #000000 !important;
    border-bottom: 1px solid #1a1a1a !important;
    padding: 0 !important;
    margin-bottom: 32px !important;
}

.tab-nav button {
    background: transparent !important;
    color: #555555 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 14px 24px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    transition: all 0.2s !important;
    border-radius: 0 !important;
}

.tab-nav button:hover {
    color: #ff2d78 !important;
}

.tab-nav button.selected {
    color: #ff2d78 !important;
    border-bottom: 2px solid #ff2d78 !important;
    background: transparent !important;
}

input[type="text"], textarea, .gr-textbox textarea {
    background: #0d0d0d !important;
    border: 1px solid #222222 !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 14px !important;
    padding: 12px 16px !important;
    transition: border-color 0.2s !important;
}

input[type="text"]:focus, textarea:focus {
    border-color: #ff2d78 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(255,45,120,0.1) !important;
}

label, .gr-label {
    color: #888888 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    margin-bottom: 6px !important;
}

button.primary, .gr-button-primary {
    background: #ff2d78 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    padding: 12px 28px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    transition: all 0.2s !important;
    box-shadow: 0 0 20px rgba(255,45,120,0.3) !important;
}

button.primary:hover {
    background: #ff0060 !important;
    box-shadow: 0 0 30px rgba(255,45,120,0.5) !important;
    transform: translateY(-1px) !important;
}

.log-box textarea {
    background: #0a0a0a !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 10px !important;
    color: #ff85b3 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    line-height: 1.8 !important;
}

.output-panel {
    background: #0d0d0d !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 12px !important;
    padding: 20px !important;
}

.gr-markdown {
    background: #0d0d0d !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 12px !important;
    padding: 20px !important;
    color: #ffffff !important;
    line-height: 1.8 !important;
}

.gr-markdown h2 {
    color: #ff2d78 !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    margin-bottom: 12px !important;
    padding-bottom: 8px !important;
    border-bottom: 1px solid #1a1a1a !important;
}

.gr-markdown p, .gr-markdown li {
    color: #cccccc !important;
    font-size: 14px !important;
}

.gr-markdown code {
    background: #1a1a1a !important;
    color: #ff85b3 !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.gr-image {
    border: 1px solid #1a1a1a !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

input[type=range] { accent-color: #ff2d78 !important; }

input[type=number] {
    background: #0d0d0d !important;
    color: #ffffff !important;
    border: 1px solid #222222 !important;
    border-radius: 10px !important;
}

table { width: 100%; border-collapse: collapse; }
th {
    background: #ff2d78 !important;
    color: #ffffff !important;
    padding: 10px 14px !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
td {
    border: 1px solid #1a1a1a !important;
    padding: 9px 14px !important;
    color: #cccccc !important;
    font-size: 13px !important;
}
tr:nth-child(even) td { background: #0d0d0d !important; }
tr:nth-child(odd) td { background: #000000 !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #000000; }
::-webkit-scrollbar-thumb { background: #ff2d78; border-radius: 4px; }

.gr-markdown:empty { display: none !important; }
.gr-textbox:empty { display: none !important; }

footer { display: none !important; }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(css=CSS, title="AI Video Analyzer") as demo:
        gr.Markdown(
            """
# AI Video Analyzer
### Multimodal understanding for live streams and recorded videos
            """
        )

        with gr.Tabs():
            with gr.TabItem("YouTube Video"):
                gr.Markdown("Paste a YouTube URL and click Analyze. The log shows each step as it completes.")

                with gr.Row():
                    yt_url = gr.Textbox(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=...", scale=4)
                    yt_btn = gr.Button("Analyze", variant="primary", scale=1)

                yt_log = gr.Textbox(label="Processing Log", lines=5, interactive=False, value="", elem_classes=["log-box"])

                with gr.Row():
                    yt_summary = gr.Markdown(label="Summary")
                    yt_kp = gr.Markdown(label="Key Points")

                yt_events = gr.Markdown(label="Topics and Events")

                with gr.Row():
                    yt_frames = gr.Image(label="Keyframes", type="pil")
                    yt_dip = gr.Image(label="DIP Analysis", type="pil")

                yt_tx = gr.Textbox(label="Transcript", lines=6, interactive=False, elem_classes=["log-box"])

                gr.Markdown("### Ask a question about this video")
                with gr.Row():
                    yt_q = gr.Textbox(label="Question", scale=4)
                    yt_ask = gr.Button("Ask", scale=1)
                yt_ans = gr.Textbox(label="Answer", interactive=False)

                yt_btn.click(
                    fn=yt_analyze,
                    inputs=[yt_url],
                    outputs=[yt_summary, yt_kp, yt_events, yt_frames, yt_dip, yt_tx, yt_log],
                )
                yt_ask.click(fn=ask_question, inputs=[yt_q], outputs=[yt_ans])

            with gr.TabItem("Live Stream"):
                gr.Markdown("Paste a YouTube Live URL. Set how many seconds to capture then click Start.")

                live_url = gr.Textbox(label="YouTube Live URL", placeholder="https://www.youtube.com/live/...")

                with gr.Row():
                    live_dur_slider = gr.Slider(minimum=30, maximum=300, value=60, step=30, label="Capture Duration (seconds)", scale=3)
                    live_dur_num = gr.Number(value=60, label="Exact seconds", minimum=10, maximum=600, precision=0, scale=1)

                live_dur_slider.change(fn=lambda value: value, inputs=[live_dur_slider], outputs=[live_dur_num])
                live_dur_num.change(fn=lambda value: value, inputs=[live_dur_num], outputs=[live_dur_slider])

                live_btn = gr.Button("Start Live Analysis", variant="primary")

                live_log = gr.Textbox(label="Processing Log", lines=5, interactive=False, value="", elem_classes=["log-box"])

                with gr.Row():
                    live_summary = gr.Markdown(label="Summary")
                    live_kp = gr.Markdown(label="Key Points")

                live_events = gr.Markdown(label="Topics and Events")

                with gr.Row():
                    live_frames = gr.Image(label="Live Frames", type="pil")
                    live_dip = gr.Image(label="DIP Analysis", type="pil")

                live_tx = gr.Textbox(label="Transcript", lines=6, interactive=False, elem_classes=["log-box"])

                gr.Markdown("### Ask a question about this stream")
                with gr.Row():
                    live_q = gr.Textbox(label="Question", scale=4)
                    live_ask = gr.Button("Ask", scale=1)
                live_ans = gr.Textbox(label="Answer", interactive=False)

                live_btn.click(
                    fn=live_analyze,
                    inputs=[live_url, live_dur_num],
                    outputs=[live_summary, live_kp, live_events, live_frames, live_dip, live_tx, live_log],
                )
                live_ask.click(fn=ask_question, inputs=[live_q], outputs=[live_ans])

            with gr.TabItem("DIP Playground"):
                gr.Markdown("Upload any image to see all 13 DIP techniques applied instantly.")

                with gr.Row():
                    dip_in = gr.Image(label="Input Image", type="pil", scale=1)
                    dip_out = gr.Image(label="DIP Pipeline Output", type="pil", scale=2)

                dip_btn = gr.Button("Run DIP Pipeline", variant="primary")

                dip_btn.click(fn=lambda image: make_dip_grid(image) if image else None, inputs=[dip_in], outputs=[dip_out])
                dip_in.change(fn=lambda image: make_dip_grid(image) if image else None, inputs=[dip_in], outputs=[dip_out])

                gr.Markdown(
                    """
## Techniques Applied

| Number | Technique | Purpose |
|---|---|---|
| 1 | Original | Reference frame |
| 2 | Grayscale | Luminance extraction |
| 3 | Gaussian Blur | Noise suppression |
| 4 | Canny Edges | Boundary detection |
| 5 | Sobel Gradient | Edge magnitude |
| 6 | Laplacian | Focus and sharpness metric |
| 7 | Histogram EQ | Global contrast boost |
| 8 | CLAHE | Adaptive local contrast |
| 9 | Adaptive Threshold | Text binarization |
| 10 | Morphological Close | Gap filling |
| 11 | Harris Corners | Feature point detection |
| 12 | DFT Spectrum | Frequency domain analysis |
| 13 | K-Means | Color segmentation |
                    """
                )

    return demo
