"""Codebook support.

A codebook is available when either

* the dataset file itself carries metadata (Stata/SPSS variable labels, value labels, file label), or
* a sidecar file sits next to it: ``<name>.codebook.*`` / ``<name>_codebook.*`` / ``codebook.*`` /
  ``<name>.dct`` / ``<name>.do`` (``.txt .md .csv .xlsx .pdf .html .json .docx`` are all accepted).

:func:`build_codebook_text` renders everything available as plain text for the in-app viewer.
"""
from __future__ import annotations

from pathlib import Path

SIDECAR_EXTS = (".csv", ".txt", ".md", ".xlsx", ".pdf", ".html", ".htm", ".json", ".docx", ".dct", ".do", ".xml", ".rtf")


def find_sidecar(data_path: Path) -> Path | None:
    stem, d = data_path.stem, data_path.parent
    names = [f"{stem}.codebook", f"{stem}_codebook", f"{stem}-codebook", f"{stem} codebook",
             f"codebook_{stem}", "codebook", "Codebook", "CODEBOOK", f"{stem}.dict", f"{stem}_dictionary", "data_dictionary"]
    for n in names:
        for e in SIDECAR_EXTS:
            p = d / (n + e)
            if p.exists():
                return p
    return None


def attach_codebook(ds, data_path: Path):
    side = find_sidecar(data_path)
    if side:
        ds.codebook_path = str(side)
    has_meta = any(c.label or c.value_labels for c in ds.columns) or bool(ds.file_label)
    ds.codebook = {"embedded": has_meta, "sidecar": str(side) if side else ""}


def has_codebook(ds) -> bool:
    return bool(ds.codebook.get("embedded") or ds.codebook.get("sidecar"))


def _read_text_sidecar(p: Path, limit: int = 400_000) -> str:
    ext = p.suffix.lower()
    try:
        if ext in (".txt", ".md", ".csv", ".do", ".dct", ".json", ".xml", ".html", ".htm", ".rtf"):
            return p.read_text(encoding="utf-8", errors="replace")[:limit]
        if ext == ".xlsx":
            import polars as pl
            out = []
            for name, df in pl.read_excel(str(p), sheet_id=0).items():
                out.append(f"--- sheet: {name} ---\n{df.head(2000)}")
            return "\n".join(out)
        if ext == ".docx":
            import zipfile, re
            with zipfile.ZipFile(p) as z:
                xml = z.read("word/document.xml").decode("utf8", "replace")
            return re.sub(r"<[^>]+>", " ", xml.replace("</w:p>", "\n"))[:limit]
        if ext == ".pdf":
            return f"(PDF codebook - open it with your PDF viewer)\n{p}"
    except Exception as e:  # noqa: BLE001
        return f"(could not read {p.name}: {e})"
    return f"(unsupported codebook format) {p}"


def build_codebook_text(ds, engine=None, max_values: int = 15) -> str:
    """Plain-text codebook: file info, per-variable type/label/value labels/summary, then any sidecar text."""
    lines = [f"CODEBOOK - {ds.name}", "=" * 72]
    if ds.file_label:
        lines.append(f"Dataset label : {ds.file_label}")
    lines += [f"Source        : {ds.source}", f"Observations  : {ds.n_rows:,}", f"Variables     : {len(ds.columns)}", ""]
    stats = engine.numeric_summary(ds.name, ds.numeric_names[:200]) if engine else {}
    for c in ds.columns:
        lines.append(f"{c.name}  [{c.sql_type}]")
        if c.label:
            lines.append(f"    label : {c.label}")
        s = stats.get(c.name)
        if s and s["n"]:
            lines.append(f"    n={s['n']:,}  mean={s['mean']:.6g}  sd={(s['sd'] if s['sd'] is not None else float('nan')):.6g}  "
                         f"min={s['min']:.6g}  median={s['median']:.6g}  max={s['max']:.6g}")
        if c.value_labels:
            lines.append("    values:")
            for i, (k, v) in enumerate(c.value_labels.items()):
                if i >= max_values:
                    lines.append(f"      ... {len(c.value_labels) - max_values} more")
                    break
                lines.append(f"      {k} = {v}")
        lines.append("")
    if ds.codebook_path:
        lines += ["", "=" * 72, f"ATTACHED CODEBOOK FILE: {ds.codebook_path}", "=" * 72,
                  _read_text_sidecar(Path(ds.codebook_path))]
    return "\n".join(lines)
