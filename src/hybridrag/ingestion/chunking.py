"""Structure-aware document chunking.

Splits a markdown document into self-contained chunks: sections are cut at
headings, paragraphs are packed greedily up to a token target, consecutive
chunks repeat a small overlap so context does not die at the boundary, and
fenced code blocks are never split.

Token sizes are *estimates* (about 4 characters per token): exact counts
need the target model's tokenizer, which would add a heavyweight, sometimes
networked dependency. The estimate is a public seam (:func:`estimate_tokens`)
so it can be swapped for a real tokenizer without touching the algorithm.
"""

import math
import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+\S")
_FENCE_MARKERS = ("```", "~~~")


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Estimate the token count of ``text`` (see module docstring)."""
    return max(1, math.ceil(len(text) / chars_per_token))


@dataclass(frozen=True, slots=True)
class Chunk:
    """A self-contained piece of a document, ready to be embedded."""

    ordinal: int  # position within the document, 0-based
    content: str


class Chunker:
    """Greedy, heading-aware chunker with paragraph overlap.

    Sizing knobs are explicit so the ingestion pipeline can drive them from
    configuration. ``overlap_tokens`` must be smaller than ``target_tokens``.
    """

    def __init__(
        self,
        target_tokens: int = 512,
        overlap_tokens: int = 64,
        chars_per_token: float = 4.0,
    ) -> None:
        if overlap_tokens >= target_tokens:
            msg = "overlap_tokens must be smaller than target_tokens"
            raise ValueError(msg)
        if chars_per_token <= 0:
            msg = "chars_per_token must be positive"
            raise ValueError(msg)
        self._target_tokens = target_tokens
        self._overlap_tokens = overlap_tokens
        self._chars_per_token = chars_per_token

    def chunk(self, text: str) -> list[Chunk]:
        """Split ``text`` into ordinal-numbered chunks."""
        contents: list[str] = []
        for heading, paragraphs in self._sections(text):
            for body in self._pack(paragraphs):
                # Every chunk of a section carries its heading as a breadcrumb:
                # an embedded chunk must be understandable on its own.
                contents.append(f"{heading}\n\n{body}" if heading else body)
        return [Chunk(ordinal=i, content=c) for i, c in enumerate(contents) if c.strip()]

    def _tokens(self, text: str) -> int:
        return max(1, math.ceil(len(text) / self._chars_per_token))

    def _sections(self, text: str) -> list[tuple[str | None, list[str]]]:
        """Walk the document once, returning ``(heading, paragraphs)`` sections.

        A single state machine handles headings *and* fenced code blocks:
        a ``#`` line inside a fence is code, not a heading, and blank lines
        inside a fence do not break the block into paragraphs. Text before
        the first heading forms a section with ``heading=None``.
        """
        sections: list[tuple[str | None, list[str]]] = []
        heading: str | None = None
        paragraphs: list[str] = []
        buffer: list[str] = []
        in_fence = False

        def flush_paragraph() -> None:
            nonlocal buffer
            if buffer and any(line.strip() for line in buffer):
                paragraphs.append("\n".join(buffer).strip())
            buffer = []

        def flush_section() -> None:
            nonlocal paragraphs
            flush_paragraph()
            if heading is not None or paragraphs:
                sections.append((heading, paragraphs))
            paragraphs = []

        for line in text.splitlines():
            stripped = line.strip()
            starts_fence = stripped.startswith(_FENCE_MARKERS)
            if in_fence:
                buffer.append(line)
                if starts_fence:
                    in_fence = False
                    flush_paragraph()
            elif starts_fence:
                flush_paragraph()
                buffer = [line]
                in_fence = True
            elif _HEADING_RE.match(stripped):
                flush_paragraph()
                flush_section()
                heading = stripped
            elif not stripped:
                flush_paragraph()
            else:
                buffer.append(line)

        flush_section()
        return sections

    def _pack(self, paragraphs: list[str]) -> list[str]:
        """Greedily pack paragraphs into chunks of at most the token target."""
        packed: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for paragraph in paragraphs:
            size = self._tokens(paragraph)
            if size > self._target_tokens:
                if current:
                    packed.append("\n\n".join(current))
                    current, current_tokens = [], 0
                packed.extend(self._split_oversized(paragraph))
                continue
            if current and current_tokens + size > self._target_tokens:
                packed.append("\n\n".join(current))
                current = self._overlap_tail(current)
                current_tokens = sum(self._tokens(p) for p in current)
            current.append(paragraph)
            current_tokens += size
        if current:
            packed.append("\n\n".join(current))
        return packed

    def _overlap_tail(self, packed: list[str]) -> list[str]:
        """Trailing paragraphs to repeat at the start of the next chunk."""
        if self._overlap_tokens <= 0:
            return []
        tail: list[str] = []
        total = 0
        for paragraph in reversed(packed):
            size = self._tokens(paragraph)
            if tail and total + size > self._overlap_tokens:
                break
            tail.insert(0, paragraph)
            total += size
        return tail

    def _split_oversized(self, paragraph: str) -> list[str]:
        """Split a paragraph larger than the target into word windows.

        Degradation path for a single block (a huge table, a fenced blob)
        that cannot respect the target: internal line structure is lost,
        only the word sequence (plus overlap) is preserved.
        """
        windows: list[str] = []
        words: list[str] = []
        words_tokens = 0
        for word in paragraph.split():
            size = self._tokens(f"{word} ")
            if words and words_tokens + size > self._target_tokens:
                windows.append(" ".join(words))
                tail: list[str] = []
                tail_tokens = 0
                for previous in reversed(words):
                    previous_size = self._tokens(f"{previous} ")
                    if tail and tail_tokens + previous_size > self._overlap_tokens:
                        break
                    tail.insert(0, previous)
                    tail_tokens += previous_size
                words, words_tokens = tail, tail_tokens
            words.append(word)
            words_tokens += size
        if words:
            windows.append(" ".join(words))
        return windows
