"""Token counting.

The ledger allocates a *budget*, so counting must be cheap and callable
thousands of times per run. Two implementations:

* ``HeuristicCounter`` — no dependencies, no network. CJK characters are ~1
  token each while Latin text is ~4 characters per token, so a single
  chars/4 rule underestimates Chinese filings by roughly 3x. Since this project
  processes both 10-K (English) and A-share annual reports (Chinese), the split
  matters: an underestimate means the ledger hands out a budget that overflows
  the real window.
* ``TiktokenCounter`` — exact for OpenAI-family tokenizers when the package and
  its BPE files are present locally. Never required.

Both are deliberately *conservative* (round up), because over-reserving wastes
budget while under-reserving truncates mid-prompt.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Protocol

_CJK = re.compile(
    r"[　-〿぀-ヿ㐀-䶿一-鿿豈-﫿＀-￯]"
)


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class HeuristicCounter:
    name = "heuristic"

    # Latin-ish text: ~4 chars/token. CJK: ~1 token/char (often 1-2).
    LATIN_CHARS_PER_TOKEN = 3.6
    CJK_TOKENS_PER_CHAR = 1.0

    def count(self, text: str) -> int:
        if not text:
            return 0
        cjk = len(_CJK.findall(text))
        other = len(text) - cjk
        est = cjk * self.CJK_TOKENS_PER_CHAR + other / self.LATIN_CHARS_PER_TOKEN
        return max(1, math.ceil(est))


class TiktokenCounter:
    name = "tiktoken"

    def __init__(self, encoding: str = "cl100k_base") -> None:
        import tiktoken  # noqa: PLC0415 - optional dependency

        self._enc = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))


@lru_cache(maxsize=1)
def default_counter() -> TokenCounter:
    """Prefer exact counting when available, fall back silently."""
    try:
        return TiktokenCounter()
    except Exception:  # noqa: BLE001 - any import/network failure means fall back
        return HeuristicCounter()


def count_tokens(text: str, counter: TokenCounter | None = None) -> int:
    return (counter or default_counter()).count(text)
