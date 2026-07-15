"""CLI для LegionAgent (S18): one-shot / streaming / интерактивный REPL."""
import argparse
import asyncio
from typing import Any, Dict

from .config import AgentConfig
from .legion_agent import LegionAgent


def build_agent(args: argparse.Namespace) -> LegionAgent:
    return LegionAgent(AgentConfig(
        model=args.model, rag=args.rag, tools=args.tools, session_id=args.session))


def chat_once(agent: LegionAgent, message: str) -> Dict[str, Any]:
    return agent.chat(message)


async def stream_once(agent: LegionAgent, message: str) -> str:
    parts = []
    async for chunk in agent.stream(message):
        print(chunk, end="", flush=True)
        parts.append(chunk)
    print()
    return "".join(parts)


async def _repl(agent: LegionAgent, stream: bool) -> None:
    print("LegionAgent REPL (пустая строка / 'exit' — выход)")
    while True:
        try:
            msg = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg or msg.lower() in ("exit", "quit"):
            break
        if stream:
            await stream_once(agent, msg)
        else:
            print(chat_once(agent, msg)["content"])
    await agent.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LegionAgent CLI")
    ap.add_argument("message", nargs="?", help="one-shot message (omit for --interactive)")
    ap.add_argument("--model", default="smart")
    ap.add_argument("--rag", action="store_true")
    ap.add_argument("--tools", action="store_true")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--session", default="cli")
    args = ap.parse_args(argv)

    agent = build_agent(args)
    if args.interactive:
        asyncio.run(_repl(agent, args.stream))
    elif args.message:
        if args.stream:
            asyncio.run(stream_once(agent, args.message))
        else:
            print(chat_once(agent, args.message)["content"])
    else:
        ap.error("provide a message or --interactive")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
