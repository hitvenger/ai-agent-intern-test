from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple, Union
import yaml

from app.models import DocumentMetadata, PolicyChunk


FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
HEADER_PATTERN = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


def parse_frontmatter_and_body(raw_text: str) -> Tuple[dict, str]:
    """Extracts YAML frontmatter and body from markdown text."""
    match = FRONTMATTER_PATTERN.match(raw_text)
    if match:
        yaml_content = match.group(1)
        body = raw_text[match.end():]
        try:
            metadata_dict = yaml.safe_load(yaml_content) or {}
        except Exception:
            metadata_dict = {}
        return metadata_dict, body
    return {}, raw_text


class KnowledgeBaseLoader:
    """
    Loads and parses knowledge-base markdown documents into granular chunks
    with front-matter metadata and citation breadcrumbs.
    """

    def __init__(self, kb_dir: Union[str, Path] = "knowledge-base"):
        self.kb_dir = Path(kb_dir)

    def load_documents(self) -> List[PolicyChunk]:
        """Loads all markdown files in the knowledge base directory and chunks them by headers."""
        if not self.kb_dir.exists():
            raise FileNotFoundError(f"Knowledge base directory not found at {self.kb_dir}")

        all_chunks: List[PolicyChunk] = []
        md_files = sorted(self.kb_dir.glob("*.md"))

        for file_path in md_files:
            file_name = file_path.name
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            raw_meta, body = parse_frontmatter_and_body(content)
            
            # Normalize metadata with defaults
            metadata = DocumentMetadata(
                document_id=str(raw_meta.get("document_id", file_path.stem)),
                title=str(raw_meta.get("title", file_path.stem)),
                status=str(raw_meta.get("status", "active")).lower(),
                effective_date=str(raw_meta.get("effective_date")) if raw_meta.get("effective_date") else None,
                superseded_date=str(raw_meta.get("superseded_date")) if raw_meta.get("superseded_date") else None,
                last_reviewed=str(raw_meta.get("last_reviewed")) if raw_meta.get("last_reviewed") else None,
                audience=str(raw_meta.get("audience", "customer")).lower(),
                policy_authority=str(raw_meta.get("policy_authority", "official")).lower(),
                supersedes=str(raw_meta.get("supersedes")) if raw_meta.get("supersedes") else None,
                superseded_by=str(raw_meta.get("superseded_by")) if raw_meta.get("superseded_by") else None,
                customer_answering=bool(raw_meta.get("customer_answering", True))
            )

            # Chunk body by headers
            file_chunks = self._chunk_markdown(file_name, body, metadata)
            all_chunks.extend(file_chunks)

        return all_chunks

    def _chunk_markdown(self, file_name: str, body: str, metadata: DocumentMetadata) -> List[PolicyChunk]:
        """Splits markdown body by level-1 and level-2 headers."""
        lines = body.splitlines()
        chunks: List[PolicyChunk] = []

        current_heading = metadata.title
        current_lines: List[str] = []
        chunk_idx = 0

        for line in lines:
            header_match = re.match(r"^(#{1,2})\s+(.+)$", line.strip())
            if header_match:
                # Save previous section if non-empty
                content = "\n".join(current_lines).strip()
                if content:
                    citation = f"{file_name}#{current_heading}"
                    chunks.append(PolicyChunk(
                        chunk_id=f"{file_name}_{chunk_idx}",
                        file_name=file_name,
                        heading=current_heading,
                        citation=citation,
                        content=content,
                        metadata=metadata
                    ))
                    chunk_idx += 1
                current_heading = header_match.group(2).strip()
                current_lines = [line]
            else:
                current_lines.append(line)

        # Flush final chunk
        content = "\n".join(current_lines).strip()
        if content:
            citation = f"{file_name}#{current_heading}"
            chunks.append(PolicyChunk(
                chunk_id=f"{file_name}_{chunk_idx}",
                file_name=file_name,
                heading=current_heading,
                citation=citation,
                content=content,
                metadata=metadata
            ))

        return chunks
