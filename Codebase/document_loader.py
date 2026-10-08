"""Resume / job-description file readers (PDF, DOCX, TXT, MD).

PDF text comes from `pypdf`; DOCX is read directly from its XML (no extra
dependency). Structuring of the extracted text is left to the LLM (extractor.py).
"""
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _clean(text):
    """Normalise whitespace while keeping line breaks (useful for quoting evidence)."""
    text = text.replace("\r", "\n").replace("\x00", " ")
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def _read_pdf(path):
    """Extract text from every page of a PDF."""
    try:
        from pypdf import PdfReader
    except ImportError:
        sys.exit("pypdf is required for PDF input: pip install pypdf")
    return "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)


def _read_docx(path):
    """Extract paragraph text (including table cells) from a .docx file."""
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    lines = []
    for para in root.iter(f"{_W_NS}p"):
        parts = []
        for node in para.iter():
            if node.tag == f"{_W_NS}t" and node.text:
                parts.append(node.text)
            elif node.tag in (f"{_W_NS}tab", f"{_W_NS}br"):
                parts.append(" ")
        lines.append("".join(parts))
    return "\n".join(lines)


def read_document(path):
    """Return cleaned plain text of a supported document (raises ValueError otherwise)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        raw = _read_pdf(path)
    elif suffix == ".docx":
        raw = _read_docx(path)
    elif suffix in (".txt", ".md"):
        raw = path.read_text(encoding="utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: {path.name}")
    return _clean(raw)


def load_resumes(directory):
    """Read all supported resumes in `directory` into {file_name: text}.

    Unreadable or empty files (e.g. scanned PDFs without a text layer) are skipped
    with a warning so a single bad file never aborts the whole ranking run.
    """
    folder = Path(directory)
    if not folder.is_dir():
        sys.exit(f"CV directory not found: {directory}")
    resumes = {}
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS or path.name.startswith("."):
            continue
        try:
            text = read_document(path)
        except Exception as err:  # corrupt file: report and continue
            print(f"[warn] skipped {path.name}: {err}", file=sys.stderr)
            continue
        if len(text) < 50:
            print(f"[warn] skipped {path.name}: no extractable text", file=sys.stderr)
            continue
        resumes[path.name] = text
    if not resumes:
        sys.exit(f"No readable resumes (.pdf/.docx/.txt/.md) found in {directory}")
    return resumes
