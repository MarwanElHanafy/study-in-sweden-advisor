"""Chat with the advisor in the terminal (for testing before the UI exists).

Usage:
  python src/chat_cli.py              # agent with tools (RAG)
  python src/chat_cli.py --baseline   # same model, no tools, no data
Type 'quit' to exit.
"""

from __future__ import annotations

import argparse

from agent import DEFAULT_MODEL, load_advisor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true", help="no tools, model knowledge only")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    advisor = load_advisor(use_tools=not args.baseline, model=args.model)
    session = advisor.new_session()
    print(f"Model: {args.model} | tools: {not args.baseline}. Type 'quit' to exit.\n")
    while True:
        try:
            msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg.lower() in {"quit", "exit"}:
            break
        if not msg:
            continue
        r = session.send(msg)
        for tc in r.tool_calls:
            print(f"  [tool] {tc['name']}({tc['args']}) -> {tc['n_results'] if tc['n_results'] is not None else 'ok'} in {tc['ms']} ms")
        print(f"\nkompass> {r.answer}\n")
        print(f"  ({r.latency_s}s, {r.llm_calls} LLM calls, {r.input_tokens}+{r.output_tokens} tokens)\n")


if __name__ == "__main__":
    main()
