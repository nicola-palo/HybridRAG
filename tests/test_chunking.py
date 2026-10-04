from itertools import pairwise

from hybridrag.ingestion.chunking import Chunker, estimate_tokens


def test_short_text_is_a_single_chunk() -> None:
    chunks = Chunker().chunk("hello world")

    assert [c.content for c in chunks] == ["hello world"]
    assert [c.ordinal for c in chunks] == [0]


def test_empty_text_yields_no_chunks() -> None:
    assert Chunker().chunk("") == []
    assert Chunker().chunk("   \n\n  ") == []


def test_chunks_never_exceed_the_target() -> None:
    chunker = Chunker(target_tokens=25, overlap_tokens=0)
    text = "\n\n".join(" ".join(["word"] * 20) for _ in range(5))  # 25-token paragraphs

    chunks = chunker.chunk(text)

    assert len(chunks) == 5
    assert all(estimate_tokens(c.content) == 25 for c in chunks)


def test_consecutive_chunks_share_the_overlap() -> None:
    chunker = Chunker(target_tokens=50, overlap_tokens=10)
    paragraphs = [" ".join(["word"] * 20) for _ in range(6)]  # 25 tokens each

    chunks = chunker.chunk("\n\n".join(paragraphs))

    assert len(chunks) == 5
    for previous, following in pairwise(chunks):
        assert following.content.split("\n\n")[0] == previous.content.split("\n\n")[-1]
    assert [c.ordinal for c in chunks] == list(range(5))


def test_oversized_paragraph_is_split_with_word_overlap() -> None:
    chunker = Chunker(target_tokens=20, overlap_tokens=8)
    paragraph = " ".join(["word"] * 40)  # 50 estimated tokens, one single block

    chunks = chunker.chunk(paragraph)

    assert len(chunks) >= 2
    words_per_chunk = [c.content.split() for c in chunks]
    assert all(len(words) <= 10 for words in words_per_chunk)
    for previous, following in pairwise(words_per_chunk):
        assert following[:4] == previous[-4:]  # 4-word overlap between windows


def test_headings_breadcrumb_sections_and_ordinals() -> None:
    text = "preamble line.\n\n## Alpha\n\nalpha body.\n\n## Beta\n\nbeta body."

    chunks = Chunker(target_tokens=512, overlap_tokens=0).chunk(text)

    assert [c.content for c in chunks] == [
        "preamble line.",
        "## Alpha\n\nalpha body.",
        "## Beta\n\nbeta body.",
    ]
    assert [c.ordinal for c in chunks] == [0, 1, 2]


def test_fenced_code_blocks_stay_atomic() -> None:
    fence = "```python\ndef f():\n    x = 1\n\n    y = 2\n\n    return x + y\n```"
    text = f"intro.\n\n{fence}\n\noutro."

    chunks = Chunker(target_tokens=16, overlap_tokens=0).chunk(text)

    assert [c.content for c in chunks] == ["intro.", fence, "outro."]


def test_headings_inside_fences_are_not_headings() -> None:
    text = "## Real\n\n```text\n# not a heading\n```\n\nafter."

    chunks = Chunker(target_tokens=512, overlap_tokens=0).chunk(text)

    # One single section: the '#' line inside the fence opened no section,
    # so the breadcrumb appears exactly once.
    assert [c.content for c in chunks] == ["## Real\n\n```text\n# not a heading\n```\n\nafter."]
