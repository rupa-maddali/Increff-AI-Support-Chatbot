from collections import deque
from typing import Tuple, Deque, List

class ConversationMemory:
    """
    Keeps the last `max_turns` user–bot pairs, returns a formatted
    string for prompt injection.
    """
    def __init__(self, max_turns: int = 10):
        self.max_turns: Deque[Tuple[str, str]] = deque(maxlen=max_turns)

    def add(self, user: str, bot: str):
        self.max_turns.append((user, bot))

    def format(self) -> str:
        out = []
        for u, b in self.max_turns:
            out.append(f"User: {u}\nBot: {b}")
        return "\n".join(out) + ("\n" if out else "")
