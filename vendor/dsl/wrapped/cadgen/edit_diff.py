"""
Line-by-line comparison of two texts.

- **unified diff**: :func:`git_unified_diff`; applied with :func:`apply_unified_diff` / :func:`parse_unified_diff`.
- Structured parsing: :func:`line_edits`; before/after pairs: :func:`line_replace_pairs` / :func:`apply_line_replace_pairs`.
- Sequential replacements by a unique ``old_string``: :func:`replace_edits` / :func:`apply_replace_edits`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher, unified_diff
from enum import Enum


class EditKind(str, Enum):
    EQUAL = "equal"
    DELETE = "delete"
    INSERT = "insert"
    REPLACE = "replace"


@dataclass(frozen=True)
class LineEdit:
    """Line ranges [start, end) as in ``difflib.SequenceMatcher.get_opcodes``."""

    kind: EditKind
    s1_start: int
    s1_end: int
    s2_start: int
    s2_end: int

    def lines_removed(self, lines_s1: list[str]) -> list[str]:
        if self.kind not in (EditKind.DELETE, EditKind.REPLACE):
            return []
        return lines_s1[self.s1_start : self.s1_end]

    def lines_added(self, lines_s2: list[str]) -> list[str]:
        if self.kind not in (EditKind.INSERT, EditKind.REPLACE):
            return []
        return lines_s2[self.s2_start : self.s2_end]


def split_lines(s: str) -> list[str]:
    return s.splitlines()


def git_unified_diff(
    s1: str,
    s2: str,
    *,
    context_lines: int = 3,
) -> str:
    """
    Unified diff (``difflib.unified_diff`` format, compatible with ``patch``).

    File names are not filled in: only ``@@`` hunks and ``-``/``+``/space lines.

    Lines come from ``splitlines()``; each patch line ends with ``\\n``.
    """
    lines1 = split_lines(s1)
    lines2 = split_lines(s2)
    # difflib adds lineterm only to headers; hunk lines (+/-/space) have no \n,
    # so "".join would glue them into one line. Normalize: every patch line ends with \n.
    parts: list[str] = []
    for chunk in unified_diff(
        lines1,
        lines2,
        fromfile="",
        tofile="",
        n=context_lines,
        lineterm="",
    ):
        parts.append(chunk if chunk.endswith("\n") else chunk + "\n")
    return "".join(parts)


_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass(frozen=True)
class UnifiedHunk:
    """One unified-diff hunk: ``old_start`` (1-based in the old file) and `` `` / ``-`` / ``+`` lines."""

    old_start: int
    lines: list[tuple[str, str]]  # (' '|'-'|'+', text without the prefix)


def parse_unified_diff(patch: str) -> list[UnifiedHunk]:
    """
    Parse a unified diff (as produced by :func:`difflib.unified_diff` / :func:`git_unified_diff`).
    Ignores ``diff --git``, ``---``, ``+++`` and ``\\ No newline...`` lines.
    """
    raw = patch.splitlines()
    i = 0
    hunks: list[UnifiedHunk] = []
    while i < len(raw):
        line = raw[i]
        if line.startswith("diff --git"):
            i += 1
            continue
        if line.startswith("---"):
            i += 1
            if i < len(raw) and raw[i].startswith("+++"):
                i += 1
            continue
        if line.startswith("@@"):
            m = _HUNK_HEADER.match(line)
            if not m:
                raise ValueError(f"invalid hunk header: {line!r}")
            old_start = int(m.group(1))
            i += 1
            body: list[tuple[str, str]] = []
            while i < len(raw):
                ln = raw[i]
                if ln.startswith("@@") or ln.startswith("---"):
                    break
                if ln.startswith("\\"):
                    i += 1
                    continue
                if not ln:
                    raise ValueError("blank line inside hunk")
                prefix = ln[0]
                if prefix not in " -+":
                    raise ValueError(f"unexpected line in hunk: {ln!r}")
                body.append((prefix, ln[1:]))
                i += 1
            hunks.append(UnifiedHunk(old_start=old_start, lines=body))
            continue
        i += 1
    return hunks


def apply_unified_diff(original: str, patch: str) -> str:
    """
    Apply a unified diff to the string ``original`` (line by line, as after ``splitlines``).

    Context and deletions are checked for a match; a mismatch raises ``ValueError``.
    Compatible with a patch from :func:`git_unified_diff`.
    """
    orig_lines = split_lines(original)
    hunks = parse_unified_diff(patch)
    out: list[str] = []
    orig_pos = 0

    for hunk in hunks:
        if hunk.old_start == 0:
            insert_at = 0
        else:
            insert_at = hunk.old_start - 1
        if insert_at < orig_pos:
            raise ValueError(
                f"hunks overlap or wrong order: expected orig_pos <= {insert_at}, got {orig_pos}"
            )
        out.extend(orig_lines[orig_pos:insert_at])
        orig_pos = insert_at

        for prefix, text in hunk.lines:
            if prefix == " ":
                if orig_pos >= len(orig_lines) or orig_lines[orig_pos] != text:
                    got = orig_lines[orig_pos] if orig_pos < len(orig_lines) else "<eof>"
                    raise ValueError(
                        f"context mismatch at old line {orig_pos + 1}: "
                        f"expected {text!r}, got {got!r}"
                    )
                out.append(text)
                orig_pos += 1
            elif prefix == "-":
                if orig_pos >= len(orig_lines) or orig_lines[orig_pos] != text:
                    got = orig_lines[orig_pos] if orig_pos < len(orig_lines) else "<eof>"
                    raise ValueError(
                        f"delete mismatch at old line {orig_pos + 1}: "
                        f"expected {text!r}, got {got!r}"
                    )
                orig_pos += 1
            else:
                out.append(text)

    out.extend(orig_lines[orig_pos:])
    return join_lines(out)


def line_edits(s1: str, s2: str, *, autojunk: bool = False) -> tuple[list[str], list[str], list[LineEdit]]:
    """
    Returns ``(lines1, lines2, edits)``.

    Each ``LineEdit`` is one block of line opcodes:
    *equal*, *delete*, *insert*, *replace*.
    """
    lines1 = split_lines(s1)
    lines2 = split_lines(s2)
    matcher = SequenceMatcher(None, lines1, lines2, autojunk=autojunk)
    edits = [
        LineEdit(EditKind(tag), i1, i2, j1, j2)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
    ]
    return lines1, lines2, edits


def line_edits_changes_only(s1: str, s2: str, *, autojunk: bool = False) -> list[LineEdit]:
    """Only the steps where something has to change (no *equal*)."""
    _, _, ops = line_edits(s1, s2, autojunk=autojunk)
    return [op for op in ops if op.kind != EditKind.EQUAL]


def line_replace_pairs(
    s1: str,
    s2: str,
    *,
    autojunk: bool = False,
    include_equal: bool = True,
) -> list[tuple[str, str]]:
    """
    List of before/after line pairs in ``SequenceMatcher.get_opcodes`` order.

    * *equal* (if ``include_equal``): ``(block, same block)``: a block from ``s1``, ``\\n`` between lines.
    * *delete*: ``(deleted block, empty string)``.
    * *insert*: ``(empty string, inserted block)``: an insertion at the current position (between the already
      processed prefix of ``original`` and the not yet read tail).
    * *replace*: ``(old block, new block)``.

    :func:`apply_line_replace_pairs` needs pairs with ``include_equal=True`` (the default),
    computed from the same ``s1`` that is passed to ``apply``.
    With ``include_equal=False`` the list is shorter (convenient for an LLM), but such a list
    **cannot** be applied reliably without the full plan.
    """
    lines1, lines2, edits = line_edits(s1, s2, autojunk=autojunk)
    pairs: list[tuple[str, str]] = []
    for op in edits:
        if op.kind == EditKind.EQUAL:
            if not include_equal:
                continue
            blk = join_lines(lines1[op.s1_start : op.s1_end])
            pairs.append((blk, blk))
        elif op.kind == EditKind.DELETE:
            blk = join_lines(lines1[op.s1_start : op.s1_end])
            pairs.append((blk, ""))
        elif op.kind == EditKind.INSERT:
            blk = join_lines(lines2[op.s2_start : op.s2_end])
            pairs.append(("", blk))
        else:
            old_b = join_lines(lines1[op.s1_start : op.s1_end])
            new_b = join_lines(lines2[op.s2_start : op.s2_end])
            pairs.append((old_b, new_b))
    return pairs


def apply_line_replace_pairs(original: str, pairs: list[tuple[str, str]]) -> str:
    """
    Apply to ``original`` the pairs from :func:`line_replace_pairs` with ``include_equal=True``.

    Walks ``original`` left to right: insertions do not advance the pointer into ``original``,
    other steps read and check the next lines of ``original``.
    At the end the whole text must be consumed.
    """
    lines = split_lines(original)
    i = 0
    out: list[str] = []
    for before, after in pairs:
        if before == "":
            out.extend(split_lines(after))
            continue
        if after == "":
            b = split_lines(before)
            n = len(b)
            if lines[i : i + n] != b:
                raise ValueError(
                    f"delete mismatch at line {i + 1}: expected {b!r}, got {lines[i : i + n]!r}"
                )
            i += n
            continue
        if before == after:
            b = split_lines(before)
            n = len(b)
            if lines[i : i + n] != b:
                raise ValueError(
                    f"equal mismatch at line {i + 1}: expected {b!r}, got {lines[i : i + n]!r}"
                )
            out.extend(b)
            i += n
            continue
        b = split_lines(before)
        n = len(b)
        if lines[i : i + n] != b:
            raise ValueError(
                f"replace mismatch at line {i + 1}: expected {b!r}, got {lines[i : i + n]!r}"
            )
        out.extend(split_lines(after))
        i += n

    if i != len(lines):
        raise ValueError(f"not all original lines consumed: remaining from line {i + 1}: {lines[i:]!r}")
    return join_lines(out)


def join_lines(lines: list[str]) -> str:
    """Join lines the way text is usually compared, without a trailing ``\\n``."""
    return "\n".join(lines)


def _ensure_single_occurrence(cur: str, old: str) -> None:
    if old == "":
        return
    n = cur.count(old)
    if n != 1:
        raise ValueError(f"old_string must occur exactly once in document, got {n}: {old!r}")


def replace_edits(before: str, after: str, *, autojunk: bool = False) -> list[tuple[str, str]]:
    """
    Two text versions -> a list of sequential replacements ``[(old_string, new_string), ...]``.

    At every step ``old_string`` occurs exactly once in the **current** text (during generation the
    context is extended upward by lines). Applied with :func:`apply_replace_edits`
    (first occurrence per step).

    The ``autojunk`` parameter is reserved for an API shared with :func:`line_edits`; the algorithm
    reduces ``cur`` to ``after`` iteratively by lines.
    """
    del autojunk
    edits: list[tuple[str, str]] = []
    cur = before
    target = after
    while cur != target:
        cl = split_lines(cur)
        tl = split_lines(target)
        i = 0
        while i < len(cl) and i < len(tl) and cl[i] == tl[i]:
            i += 1
        j = 0
        while (
            j < len(cl) - i
            and j < len(tl) - i
            and cl[len(cl) - 1 - j] == tl[len(tl) - 1 - j]
        ):
            j += 1
        end_cl = len(cl) - j
        end_tl = len(tl) - j
        old_mid = join_lines(cl[i:end_cl])
        new_mid = join_lines(tl[i:end_tl])

        if old_mid == "":
            if cur == "":
                edits.append(("", new_mid))
                cur = new_mid
                continue
            if i == len(cl):
                anchor = cl[-1]
                pair = (anchor, anchor + "\n" + new_mid)
            elif i == 0:
                anchor = cl[0]
                pair = (anchor, new_mid + "\n" + anchor)
            else:
                pair = (
                    cl[i - 1] + "\n" + cl[i],
                    cl[i - 1] + "\n" + new_mid + "\n" + cl[i],
                )
            _ensure_single_occurrence(cur, pair[0])
            edits.append(pair)
            cur = cur.replace(pair[0], pair[1], 1)
            continue

        lo = i
        while True:
            old_str = join_lines(cl[lo:end_cl])
            new_str = join_lines(tl[lo:end_tl])
            if cur.count(old_str) == 1:
                edits.append((old_str, new_str))
                cur = cur.replace(old_str, new_str, 1)
                break
            if lo == 0:
                raise ValueError(
                    f"cannot build unique old_string for replace step near line {i + 1}: {old_mid!r}"
                )
            lo -= 1
    return edits


def apply_replace_edits(original: str, edits: list[tuple[str, str]], *, unique: bool = True) -> str:
    """
    Apply a sequence of ``old_string`` / ``new_string`` pairs: at each step the **first**
    occurrence of ``old_string`` is replaced.

    With ``unique=True`` a non-empty ``old_string`` must occur exactly once in the current text.
    Empty ``old_string``: a single insertion at the start (like ``str.replace`` with an empty needle and limit 1).
    """
    cur = original
    for old_s, new_s in edits:
        if unique and old_s != "":
            _ensure_single_occurrence(cur, old_s)
        if old_s == "":
            cur = cur.replace("", new_s, 1)
        else:
            cur = cur.replace(old_s, new_s, 1)
    return cur


def reconstruct_s2(lines1: list[str], lines2: list[str], edits: list[LineEdit]) -> str:
    """Check that the result equals ``join_lines(lines2)``."""
    out: list[str] = []
    for op in edits:
        if op.kind == EditKind.EQUAL:
            out.extend(lines1[op.s1_start : op.s1_end])
        elif op.kind == EditKind.DELETE:
            continue
        elif op.kind == EditKind.INSERT:
            out.extend(lines2[op.s2_start : op.s2_end])
        elif op.kind == EditKind.REPLACE:
            out.extend(lines2[op.s2_start : op.s2_end])
    return join_lines(out)


def format_line_plan(
    s1: str,
    s2: str,
    *,
    include_equal: bool = False,
    autojunk: bool = False,
    one_based: bool = True,
) -> str:
    """Short text description of the operations (line numbers are 1-based if ``one_based``)."""

    def ln(i: int) -> int:
        return i + 1 if one_based else i

    lines1, lines2, edits = line_edits(s1, s2, autojunk=autojunk)
    parts: list[str] = []
    for op in edits:
        if op.kind == EditKind.EQUAL and not include_equal:
            continue
        if op.kind == EditKind.EQUAL:
            parts.append(
                f"keep s1 lines [{ln(op.s1_start)}:{ln(op.s1_end)}) "
                f"== s2 [{ln(op.s2_start)}:{ln(op.s2_end)})"
            )
        elif op.kind == EditKind.DELETE:
            parts.append(
                f"delete s1 lines [{ln(op.s1_start)}:{ln(op.s1_end)}): "
                + repr(lines1[op.s1_start : op.s1_end])
            )
        elif op.kind == EditKind.INSERT:
            parts.append(
                f"insert before s1 line {ln(op.s1_start)} from s2 "
                f"[{ln(op.s2_start)}:{ln(op.s2_end)}): "
                + repr(lines2[op.s2_start : op.s2_end])
            )
        else:
            parts.append(
                f"replace s1 [{ln(op.s1_start)}:{ln(op.s1_end)}) -> "
                f"s2 [{ln(op.s2_start)}:{ln(op.s2_end)}):\n"
                f"  - {repr(lines1[op.s1_start : op.s1_end])}\n"
                f"  + {repr(lines2[op.s2_start : op.s2_end])}"
            )
    return "\n".join(parts)


__all__ = [
    "EditKind",
    "LineEdit",
    "UnifiedHunk",
    "apply_replace_edits",
    "apply_line_replace_pairs",
    "apply_unified_diff",
    "replace_edits",
    "format_line_plan",
    "git_unified_diff",
    "join_lines",
    "line_edits",
    "line_edits_changes_only",
    "line_replace_pairs",
    "parse_unified_diff",
    "reconstruct_s2",
    "split_lines",
]
