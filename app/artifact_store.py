#!/usr/bin/env python3
r"""Local-first workspace artifact store for coPPeR.

Manages Markdown notes, interactive checklists, structured tables, and DOCX export.
All files are stored under a configurable base directory (default: workspace_notes/).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASE_DIR = "workspace_notes"
ARTIFACT_TYPES = ("note", "checklist", "table")
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _make_slug(title: str) -> str:
    """Convert a human-readable title to a filesystem-safe slug."""
    slug = title.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    if not _SLUG_RE.match(slug):
        slug = "untitled-" + slug[:20]
    return slug


def _artifact_path(base_dir: Path, slug: str) -> Path:
    """Return the .md file path for an artifact."""
    return base_dir / f"{slug}.md"


# ---------------------------------------------------------------------------
# ArtifactStore — CRUD + parsers/serializers
# ---------------------------------------------------------------------------


class ArtifactStore:
    """Local-first storage manager for workspace artifacts (notes, checklists, tables)."""

    def __init__(self, base_dir: str | Path = DEFAULT_BASE_DIR) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ---- Metadata helpers ----

    def _read_meta(self, slug: str) -> dict[str, Any] | None:
        """Read the JSON metadata sidecar for an artifact."""
        meta_path = self._meta_path(slug)
        if not meta_path.exists():
            return None
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None

    def _write_meta(self, slug: str, title: str, artifact_type: str = "note") -> None:
        """Write JSON metadata sidecar."""
        meta_path = self._meta_path(slug)
        meta_path.write_text(
            json.dumps({"title": title, "type": artifact_type}, ensure_ascii=False),
            encoding="utf-8",
        )

    def _meta_path(self, slug: str) -> Path:
        return self.base_dir / f"{slug}.meta.json"

    # ---- CRUD operations ----

    def list_notes(self) -> list[dict[str, Any]]:
        """List all saved artifacts with metadata."""
        results = []
        if not self.base_dir.exists():
            return results
        for meta_file in sorted(self.base_dir.glob("*.meta.json")):
            # Stem is "slug.meta" — strip the .meta suffix to get the actual slug
            raw_stem = meta_file.stem  # e.g. "alpha.meta"
            if raw_stem.endswith(".meta"):
                slug = raw_stem[:-5]
            else:
                slug = raw_stem
            md_path = _artifact_path(self.base_dir, slug)
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            size = md_path.stat().st_size if md_path.exists() else 0
            results.append(
                {
                    "slug": slug,
                    "title": meta.get("title", slug),
                    "type": meta.get("type", "note"),
                    "size_bytes": size,
                    "md_path": str(md_path),
                }
            )
        return results

    def get_note(self, slug: str) -> dict[str, Any] | None:
        """Load an artifact by slug. Returns {title, content, type} or None."""
        md_path = _artifact_path(self.base_dir, slug)
        if not md_path.exists():
            return None
        meta = self._read_meta(slug)
        content = md_path.read_text(encoding="utf-8")
        return {
            "slug": slug,
            "title": meta.get("title", slug) if meta else slug,
            "type": meta.get("type", "note") if meta else "note",
            "content": content,
        }

    def save_note(
        self,
        title: str,
        content: str = "",
        artifact_type: str = "note",
        slug_override: str | None = None,
    ) -> dict[str, Any]:
        """Save (or update) an artifact to local disk."""
        slug = slug_override or _make_slug(title)
        md_path = _artifact_path(self.base_dir, slug)
        meta_path = self._meta_path(slug)

        # Write content
        md_path.write_text(content, encoding="utf-8", newline="\n")
        # Write metadata
        if artifact_type not in ARTIFACT_TYPES:
            artifact_type = "note"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump({"title": title, "type": artifact_type}, f)

        return {"slug": slug, "title": title, "type": artifact_type}

    def delete_note(self, slug: str) -> bool:
        """Delete an artifact and its metadata sidecar."""
        md_path = _artifact_path(self.base_dir, slug)
        meta_path = self._meta_path(slug)
        deleted = False
        if md_path.exists():
            md_path.unlink()
            deleted = True
        if meta_path.exists():
            meta_path.unlink()
            deleted = True
        return deleted

    # ---- Checklist parser / toggler ----

    @staticmethod
    def parse_checklist(md_text: str) -> list[dict[str, Any] | None]:
        """Parse Markdown checklist lines into structured items.

        Returns a list of dicts with keys: checked (bool), text (str).
        Non-checklist lines are returned as None to preserve positions.
        """
        items = []
        for line in md_text.split("\n"):
            m = re.match(r"^- \[( |x|X)\](.*)$", line)
            if m:
                checked = m.group(1).lower() == "x"
                text = m.group(2).strip()
                items.append({"checked": checked, "text": text})
            else:
                items.append(None)  # non-checklist line
        return items

    @staticmethod
    def toggle_checklist_item(md_text: str, index: int) -> str:
        """Toggle the checkbox state of item at *index* (0-based among checklist lines).

        Returns updated Markdown text. Non-checklist lines are skipped when counting.
        """
        lines = md_text.split("\n")
        check_idx = 0
        for i, line in enumerate(lines):
            m = re.match(r"^(- \[)( |x|X)(\])(.*)$", line)
            if m:
                if check_idx == index:
                    current_state = m.group(2).lower()
                    new_state = " " if current_state == "x" else "x"
                    lines[i] = f"{m.group(1)}{new_state}{m.group(3)}{m.group(4)}"
                    break
                check_idx += 1
        return "\n".join(lines)

    # ---- Markdown table parser / serializer ----

    @staticmethod
    def parse_markdown_table(md_text: str) -> dict[str, Any] | None:
        """Parse a GFM Markdown table from text.

        Returns {headers: list[str], rows: list[list[str]]} or None if no valid table found.
        """
        lines = [l.strip() for l in md_text.split("\n") if l.strip()]
        # Find the separator line (---|---|---)
        sep_idx = None
        for i, line in enumerate(lines):
            if re.match(r"^\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)+\|?$", line):
                sep_idx = i
                break

        if sep_idx is None or sep_idx == 0:
            return None

        headers_line = lines[sep_idx - 1]
        header_cells = [c.strip() for c in headers_line.split("|") if c.strip()]

        rows = []
        for line in lines[sep_idx + 1 :]:
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if len(cells) == len(header_cells):
                rows.append(cells)
            elif cells:
                # Pad or truncate to match header length
                padded = cells[: len(header_cells)]
                while len(padded) < len(header_cells):
                    padded.append("")
                rows.append(padded)

        if not header_cells and not rows:
            return None

        return {"headers": header_cells, "rows": rows}

    @staticmethod
    def serialize_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
        """Serialize headers + rows into GFM Markdown table text."""
        if not headers:
            return ""
        # Build separator line
        sep = "| " + " | ".join("-" * 3 for _ in headers) + " |"
        lines = ["| " + " | ".join(headers) + " |", sep]
        for row in rows:
            padded = list(row)
            while len(padded) < len(headers):
                padded.append("")
            lines.append(
                "| " + " | ".join(str(c) for c in padded[: len(headers)]) + " |"
            )
        return "\n".join(lines)

    # ---- DOCX export ----

    @staticmethod
    def export_to_docx(
        title: str, md_text: str, out_path: str | Path
    ) -> dict[str, Any]:
        """Convert Markdown text to a .docx file.

        Handles headings (h1-h6), paragraphs, bullet lists, checklist items, and GFM tables.
        Uses python-docx; installs it if missing from build\\.venv.
        """
        out_path = Path(out_path)
        try:
            from docx import Document  # type: ignore[import-untyped]
        except ImportError:
            return {
                "error": "python-docx not installed. Install via: pip install python-docx"
            }

        doc = Document()
        # Title as heading 0 (Title style)
        doc.add_heading(title, level=0)

        md_lines = md_text.split("\n")

        for raw_line in md_lines:
            line = raw_line.rstrip()

            # Skip empty lines but allow blank paragraphs
            if not line:
                doc.add_paragraph("")
                continue

            # Headings (# to ######)
            heading_match = re.match(r"^(#{1,6})\s+(.*)", line)
            if heading_match:
                level = min(len(heading_match.group(1)), 6)
                doc.add_heading(heading_match.group(2), level=level)
                continue

            # Checklist item (- [ ] or - [x])
            check_match = re.match(r"^- \[( |x|X)\]\s+(.*)", line)
            if check_match:
                checked = check_match.group(1).lower() == "x"
                text = f"[{'x' if checked else ' '}] {check_match.group(2)}"
                doc.add_paragraph(text, style="List Bullet")
                continue

            # Bullet list item (- or *)
            bullet_match = re.match(r"^[-*]\s+(.*)", line)
            if bullet_match:
                doc.add_paragraph(bullet_match.group(1), style="List Bullet")
                continue

            # Table detection — check if this looks like a table cell or separator
            if "|" in line and any(c.strip() for c in line.split("|")):
                # Collect consecutive table lines
                table_lines = []
                start_idx = md_lines.index(raw_line) if raw_line in md_lines else 0
                for j in range(start_idx, len(md_lines)):
                    tline = md_lines[j].strip()
                    if re.match(r"^\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)+\|?$", tline):
                        # Separator line — include it and stop collecting
                        table_lines.append(tline)
                        break
                    if "|" in tline:
                        table_lines.append(tline)

                if len(table_lines) >= 3:
                    parsed = ArtifactStore.parse_markdown_table("\n".join(table_lines))
                    if parsed and parsed["headers"]:
                        num_rows = max(len(parsed["rows"]) + 1, 2)
                        table = doc.add_table(
                            rows=num_rows, cols=len(parsed["headers"])
                        )
                        for ci, h in enumerate(parsed["headers"]):
                            cell = table.rows[0].cells[ci]
                            p_elem = cell.paragraphs[0]
                            run = p_elem.add_run(h)
                            run.bold = True
                        for ri, row_data in enumerate(parsed["rows"], start=1):
                            for ci, val in enumerate(row_data):
                                if ci < len(table.rows[ri].cells):
                                    table.rows[ri].cells[ci].text = val
                        continue

            # Regular paragraph — skip pure separator lines
            if re.match(r"^\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)+\|?$", line):
                continue

            doc.add_paragraph(line)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(out_path))
        return {
            "success": True,
            "path": str(out_path),
            "size_bytes": out_path.stat().st_size,
        }
