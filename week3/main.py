"""
main.py
-------
Interactive demo of the AT&T assistant.

Run:
    python main.py

Commands inside the chat:
    /status    show which embedding backend is active and memory sizes
    /reset     start a new session (clears short-term memory, keeps long-term)
    /remember  save a fact explicitly, e.g.  /remember I am on a family plan
    /forget    wipe long-term memory
    /quit      exit

Try this to see LONG-TERM memory working:
    1. Run, say "my name is Vishnu", then /quit
    2. Run again and ask "what is my name?"  -> it still knows.

Try this to see SESSION memory working:
    1. "what internet options do you offer?"
    2. "is that available everywhere?"   -> "that" resolves from history.

Try this to see GUARDRAILS working:
    "I have a headache, what medicine should I take?"  -> refusal
"""

from chatbot import ChatBot


BANNER = """AT&T Assistant (week 3: grounding + guardrails + memory)
Type /quit to exit, /status for info, /reset for a new session.
"""


def main() -> None:
    bot = ChatBot()
    print(BANNER)
    print(bot.status(), "\n")

    if bot.embedder.backend == "hashing":
        print(
            "Note: Ollama was not reachable, so the fallback keyword-style embedder "
            "is in use.\n      For real semantic search: install Ollama and run "
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
        if query == "/status":
            print(bot.status(), "\n")
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
            print(f"Bot: {bot.get_answer(query)}\n")
        except Exception as exc:  # keep the chat alive on API hiccups
            print(f"[error] {exc}\n")


if __name__ == "__main__":
    main()
