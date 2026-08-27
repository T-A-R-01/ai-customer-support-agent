import re
from dataclasses import dataclass
from typing import Any

from app.rag.loader import Document


@dataclass
class Chunk:
    """A searchable piece of a knowledge-base document."""

    chunk_id: str
    source: str
    heading: str
    metadata: dict[str, Any]
    text: str


def split_into_sections(content: str) -> list[tuple[str, str]]:
    """
    Split Markdown content into sections based on headings.

    Returns:
        A list of (heading, section_text) pairs.
    """

    heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

    matches = list(heading_pattern.finditer(content))

    if not matches:
        return [("General", content.strip())]

    sections = []

    # Content before the first heading.
    if matches[0].start() > 0:
        intro = content[:matches[0].start()].strip()

        if intro:
            sections.append(("Introduction", intro))

    for index, match in enumerate(matches):
        heading = match.group(2).strip()

        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)

        section_text = content[start:end].strip()

        if section_text:
            sections.append((heading, section_text))

    return sections


def split_long_text(text: str, max_chars: int = 1200) -> list[str]:
    """
    Split a long section into smaller pieces.

    We prefer paragraph boundaries instead of cutting sentences
    in the middle.
    """

    if len(text) <= max_chars:
        return [text]

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:
        if not current:
            current = paragraph
            continue

        candidate = f"{current}\n\n{paragraph}"

        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)

    return chunks


def chunk_document(document: Document, max_chars: int = 1200) -> list[Chunk]:
    """
    Convert one Document into searchable chunks.
    """

    sections = split_into_sections(document.content)

    chunks = []

    for section_number, (heading, section_text) in enumerate(sections, start=1):
        text_parts = split_long_text(section_text, max_chars=max_chars)

        for part_number, text in enumerate(text_parts, start=1):
            chunk_id = (
                f"{document.source}"
                f"::section-{section_number}"
                f"::part-{part_number}"
            )

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source=document.source,
                    heading=heading,
                    metadata=document.metadata.copy(),
                    text=text,
                )
            )

    return chunks


def chunk_documents(
    documents: list[Document],
    max_chars: int = 1200,
) -> list[Chunk]:
    """
    Chunk every document in the knowledge base.
    """

    all_chunks = []

    for document in documents:
        all_chunks.extend(
            chunk_document(document, max_chars=max_chars)
        )

    return all_chunks