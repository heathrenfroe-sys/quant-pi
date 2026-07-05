import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


@dataclass
class BrainCorpus:
    text: str
    file_count: int
    char_count: int

    @property
    def approx_tokens(self) -> int:
        return self.char_count // 4


def _strip_frontmatter(md: str) -> str:
    return FRONTMATTER_RE.sub("", md, count=1)


def _extract_wikilinks(md: str) -> list[str]:
    """Return list of bare note names from [[...]] links. Strips aliases and section anchors."""
    out = []
    for raw in WIKILINK_RE.findall(md):
        # [[Note|alias]] -> Note ; [[Note#Section]] -> Note ; [[folder/Note]] -> folder/Note
        name = raw.split("|", 1)[0].split("#", 1)[0].strip()
        if name:
            out.append(name)
    return out


def _build_name_index(vault_path: Path) -> dict[str, Path]:
    """Map basename (without .md) -> path. Obsidian's default link resolution: shortest-path first match by basename."""
    idx: dict[str, Path] = {}
    for f in vault_path.rglob("*.md"):
        idx.setdefault(f.stem, f)
        # Also index by relative path with and without extension for explicit-path links
        rel = f.relative_to(vault_path).as_posix()
        idx.setdefault(rel, f)
        idx.setdefault(rel[:-3] if rel.endswith(".md") else rel, f)
    return idx


def _resolve(name: str, index: dict[str, Path]) -> Optional[Path]:
    if name in index:
        return index[name]
    # Try with .md
    if (name + ".md") in index:
        return index[name + ".md"]
    return None


def load_linked(vault_path: Path, root_note: str, max_depth: int = 3,
                max_files: int = 200) -> BrainCorpus:
    """Load the project's root note and everything reachable via [[wikilinks]] up to max_depth.

    root_note: vault-relative path or bare note name (e.g. "45 Freelance Projects/Quantberry-Pi Paper Trader" or "Quantberry-Pi Paper Trader").
    """
    if not vault_path.exists():
        raise FileNotFoundError(f"Vault path does not exist: {vault_path}")

    index = _build_name_index(vault_path)
    start = _resolve(root_note, index)
    if start is None:
        raise FileNotFoundError(f"Root note not found in vault: {root_note}")

    seen: set[Path] = set()
    parts: list[str] = []
    queue: deque[tuple[Path, int]] = deque([(start, 0)])

    while queue and len(seen) < max_files:
        path, depth = queue.popleft()
        if path in seen:
            continue
        seen.add(path)

        try:
            md = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        body = _strip_frontmatter(md).strip()
        if body:
            rel = path.relative_to(vault_path).as_posix()
            parts.append(f"### {rel}\n\n{body}")

        if depth < max_depth:
            for name in _extract_wikilinks(md):
                target = _resolve(name, index)
                if target and target not in seen:
                    queue.append((target, depth + 1))

    text = "\n\n---\n\n".join(parts)
    return BrainCorpus(text=text, file_count=len(parts), char_count=len(text))


def load_subfolder(vault_path: Path, subfolder: Optional[str] = None) -> BrainCorpus:
    root = vault_path / subfolder if subfolder else vault_path
    if not root.exists():
        raise FileNotFoundError(f"Vault path does not exist: {root}")

    parts: list[str] = []
    files = sorted(root.rglob("*.md"))
    for f in files:
        try:
            md = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        md = _strip_frontmatter(md).strip()
        if not md:
            continue
        rel = f.relative_to(root).as_posix()
        parts.append(f"### {rel}\n\n{md}")

    text = "\n\n---\n\n".join(parts)
    return BrainCorpus(text=text, file_count=len(parts), char_count=len(text))


def load_corpus(vault_path: Path, subfolder: Optional[str] = None,
                root_note: Optional[str] = None, max_depth: int = 3) -> BrainCorpus:
    """Dispatcher: prefers link-traversal from root_note if set, else subfolder, else whole vault."""
    if root_note:
        return load_linked(vault_path, root_note, max_depth=max_depth)
    return load_subfolder(vault_path, subfolder)


def main() -> None:
    import sys
    from quant_pi.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[2] / "config.toml")
    corpus = load_corpus(cfg.vault_path, cfg.vault_subfolder, cfg.brain_root_note, cfg.brain_max_depth)
    mode = "link-traversal" if cfg.brain_root_note else ("subfolder" if cfg.vault_subfolder else "whole vault")
    print(f"Mode:  {mode}")
    if cfg.brain_root_note:
        print(f"Root:  {cfg.brain_root_note}  (depth={cfg.brain_max_depth})")
    print(f"Files: {corpus.file_count}")
    print(f"Chars: {corpus.char_count:,}")
    print(f"Approx tokens: {corpus.approx_tokens:,}")
    if corpus.file_count == 0:
        print("WARNING: no markdown files found.", file=sys.stderr)
        sys.exit(1)
    print("\n--- First 500 chars ---")
    print(corpus.text[:500])


if __name__ == "__main__":
    main()
