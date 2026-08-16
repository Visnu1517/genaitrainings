"""
main.py
-------
Interactive demo of the AT&T assistant with tool calling.

Run:
    python main.py            # normal
    python main.py --verbose  # show tool selection, arguments and results

Commands:
    /tools     list the registered tools and their schemas
    /status    embedding backend, tool count, memory sizes
    /verbose   toggle tool tracing on and off
    /reset     new session (clears short-term memory, keeps long-term)
    /remember  save a fact, e.g.  /remember my account is ACC-100001
    /forget    wipe long-term memory
    /quit      exit

Prompts worth trying:

  TOOL CALLING
    Is fiber available in 75201?
    How much is Unlimited Extra for 3 lines?
    Any outages in 10001?
    What's the balance on account ACC-100001?
    Find AT&T stores near 30301

  VALIDATION FAILURE AND RECOVERY  (the model should ask you to correct it)
    What's the balance on account 12345?
    Is fiber available in ZIP 752?

  GROUNDING (answered from docs, no tool needed)
    What is AT&T Fiber?

  GUARDRAILS
    I have a headache, what medicine should I take?

  MEMORY
    my name is Vishnu        -> then /quit, restart, and ask "what is my name?"
"""

import json
import sys

from chatbot import ChatBot, all_specs


BANNER = """AT&T Assistant (week 4: tools + Pydantic validation)
Type /quit to exit, /tools to list tools, /status for info.
"""


def print_tools() -> None:
    for spec in all_specs():
        props = spec["input_schema"].get("properties", {})
        required = set(spec["input_schema"].get("required", []))
        print(f"\n  {spec['name']}")
        print(f"    {spec['description']}")
        for field, meta in props.items():
            mark = "required" if field in required else "optional"
            constraints = {
                k: v
                for k, v in meta.items()
                if k in ("pattern", "enum", "minimum", "maximum", "minLength", "maxLength")
            }
            extra = f"  {json.dumps(constraints)}" if constraints else ""
            print(f"      - {field} ({meta.get('type', '?')}, {mark}){extra}")
    print()


def main() -> None:
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    bot = ChatBot(verbose=verbose)

    print(BANNER)
    print(bot.status(), "\n")

    if bot.embedder.backend == "hashing":
        print(
            "Note: Ollama was not reachable, so the fallback keyword-style embedder is in "
            "use.\n      For real semantic search: install Ollama and run "
            "`ollama pull nomic-embed-text`.\n"
        )

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue
        if query == "/quit":
            break
        if query == "/tools":
            print_tools()
            continue
        if query == "/status":
            print(bot.status(), "\n")
            continue
        if query == "/verbose":
            bot.verbose = not bot.verbose
            print(f"(tool tracing {'on' if bot.verbose else 'off'})\n")
            continue
        if query == "/reset":
            bot.reset_session()
            print("(session cleared - long-term memory kept)\n")
            continue
        if query == "/forget":
            bot.forget()
            print("(long-term memory wiped)\n")
            continue
        if query.startswith("/remember "):
            fact = query[len("/remember "):].strip()
            if fact:
                bot.long_term.remember_fact(fact)
                print(f"(remembered: {fact})\n")
            continue

        try:
            answer = bot.get_answer(query)
            if bot.last_tool_calls and not bot.verbose:
                used = ", ".join(
                    f"{c['tool']}{'' if c['ok'] else ' (failed)'}"
                    for c in bot.last_tool_calls
                )
                print(f"  [tools used: {used}]")
            print(f"Bot: {answer}\n")
        except Exception as exc:  # keep the chat alive on API hiccups
            print(f"[error] {exc}\n")


if __name__ == "__main__":
    main()
