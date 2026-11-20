"""Kompass web demo (Gradio).

Usage:
  python src/app.py            # opens http://127.0.0.1:7860
  python src/app.py --share    # temporary public link (e.g. for a recorded or remote demo)
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import gradio as gr

from agent import APP_NAME, DEFAULT_MODEL, load_advisor

EXAMPLES = [
    "I have a bachelor's in computer science and want to work with AI. My budget is max 250,000 SEK. What can I study?",
    "Compare Uppsala and Lund for someone who wants classic student life and low rent.",
    "Is there a master's in game development outside Stockholm?",
    "How much money do I need per month for a Swedish student residence permit?",
]

INTRO = f"""### {APP_NAME}: find what and where to study in Sweden
Ask about English-taught master's programmes (2027 intakes) and Swedish study cities.
Answers are grounded in official open data from Skolverket (Susa-navet) and curated city profiles.
**Always verify details on the programme page and universityadmissions.se before applying.**"""


def dataset_note() -> str:
    try:
        meta = json.loads(Path("data/raw/manifest.json").read_text())
        n = len(json.loads(Path("data/programs.json").read_text()))
        return f"Data: {n} programme offerings, Skolverket Susa-navet, fetched {meta['fetched_at'][:10]}. Model: {DEFAULT_MODEL}."
    except Exception:
        return f"Model: {DEFAULT_MODEL}."


def trace_markdown(r) -> str:
    lines = [f"**{r.latency_s} s**, {r.llm_calls} LLM calls, {r.input_tokens:,} input / {r.output_tokens:,} output tokens", ""]
    if r.tool_calls:
        lines.append("**Tools called**")
        for tc in r.tool_calls:
            args = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
            n = f", {tc['n_results']} results" if tc["n_results"] is not None else ""
            lines.append(f"- `{tc['name']}({args})`{n}, {tc['ms']} ms")
    else:
        lines.append("No tools called (answered from the conversation).")
    if r.sources:
        lines += ["", "**Sources used**"] + [f"- [{s['title']}]({s['url']})" for s in r.sources[:12]]
    return "\n".join(lines)


def build_ui(advisor) -> gr.Blocks:
    chatbot_kwargs = {"height": 560, "label": APP_NAME}
    if "type" in inspect.signature(gr.Chatbot.__init__).parameters:  # Gradio 4/5 need this, 6 has it built in
        chatbot_kwargs["type"] = "messages"

    with gr.Blocks(title=APP_NAME) as demo:
        session = gr.State(None)
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(**chatbot_kwargs)
                box = gr.Textbox(placeholder="e.g. I studied biology and want a master's in a small student town", show_label=False)
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    reset = gr.Button("New conversation")
                gr.Examples(EXAMPLES, inputs=box)
            with gr.Column(scale=2):
                gr.Markdown("### How this answer was produced")
                trace = gr.Markdown("Ask a question to see which tools the agent used.")
        gr.Markdown(dataset_note())

        def respond(message, history, sess):
            message = (message or "").strip()
            if not message:
                return "", history, sess, gr.update()
            sess = sess or advisor.new_session()
            history = (history or []) + [{"role": "user", "content": message}]
            try:
                r = sess.send(message)
                answer, info = r.answer, trace_markdown(r)
            except Exception as e:  # keep the demo alive on API errors (e.g. rate limits)
                answer, info = f"Sorry, something went wrong ({type(e).__name__}). Please try again in a moment.", f"Error: `{e}`"
            history.append({"role": "assistant", "content": answer})
            return "", history, sess, info

        send.click(respond, [box, chat, session], [box, chat, session, trace])
        box.submit(respond, [box, chat, session], [box, chat, session, trace])
        reset.click(lambda: ([], None, "New conversation started."), None, [chat, session, trace])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build_ui(load_advisor()).launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
