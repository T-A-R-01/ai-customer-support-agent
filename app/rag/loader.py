from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Document:
    """A single knowledge-base document."""

    source: str
    metadata: dict[str, Any]
    content: str


def parse_markdown_file(path: Path) -> Document:
    """
    Read one Markdown file and separate its YAML front matter
    from the main document content.
    """

    text = path.read_text(encoding="utf-8")

    metadata: dict[str, Any] = {}
    content = text

    # Front matter is expected to be between two --- markers.
    if text.startswith("---"):
        parts = text.split("---", 2)

        if len(parts) == 3:
            _, front_matter, content = parts
            metadata = yaml.safe_load(front_matter) or {}

    return Document(
        source=path.name,
        metadata=metadata,
        content=content.strip(),
    )


def load_knowledge_base(directory: str = "knowledge-base") -> list[Document]:
    """
    Load all Markdown documents from the knowledge-base directory.
    """

    kb_path = Path(directory)

    if not kb_path.exists():
        raise FileNotFoundError(
            f"Knowledge-base directory not found: {kb_path}"
        )

    documents = []

    for path in sorted(kb_path.glob("*.md")):
        documents.append(parse_markdown_file(path))

    return documents