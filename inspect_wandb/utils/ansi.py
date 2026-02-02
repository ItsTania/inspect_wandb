"""Utilities for handling ANSI escape codes in terminal output."""

import re


# Comprehensive ANSI escape code pattern
# Matches:
# - CSI sequences: \x1b[...m (colors, styles)
# - CSI sequences: \x1b[...H, \x1b[...J, etc. (cursor movement, clearing)
# - CSI with DEC private modes: \x1b[?...h/l (show/hide cursor, etc.)
# - OSC sequences: \x1b]...BEL or \x1b]...\x1b\\ (terminal titles, hyperlinks)
# - Other escape sequences: \x1b[...
ANSI_PATTERN = re.compile(
    r"\x1b\["  # CSI escape
    r"[?]?"  # Optional DEC private mode indicator
    r"[0-9;]*"  # parameters
    r"[A-Za-z]"  # command character
    r"|"
    r"\x1b\]"  # OSC escape
    r"[^\x07\x1b]*"  # content (until BEL or ESC)
    r"(?:\x07|\x1b\\)"  # terminator (BEL or ST)
    r"|"
    r"\x1b[PX^_][^\x1b]*\x1b\\"  # DCS, SOS, PM, APC sequences
    r"|"
    r"\x1b."  # Other two-character escapes
)


def strip_ansi(text: str) -> str:
    """
    Remove ANSI escape codes from text.

    This handles common terminal escape sequences including:
    - Color codes (foreground/background colors)
    - Text styles (bold, italic, underline, etc.)
    - Cursor movement and positioning
    - Screen clearing
    - Terminal title sequences
    - Hyperlinks (OSC 8)

    Args:
        text: Input text potentially containing ANSI escape codes.

    Returns:
        Text with all ANSI escape codes removed.
    """
    return ANSI_PATTERN.sub("", text)


def clean_tui_output(text: str) -> str:
    """
    Clean up terminal UI output by removing excessive whitespace.

    This is useful after stripping ANSI codes from TUI applications
    (like Rich/Textual) that use cursor positioning, which leaves
    behind lots of empty lines and trailing spaces.

    Args:
        text: Text to clean (should already have ANSI codes stripped).

    Returns:
        Cleaned text with:
        - Trailing whitespace removed from each line
        - Lines that are only whitespace removed
        - Multiple consecutive blank lines collapsed to one
        - Leading/trailing blank lines removed
    """
    lines = text.split("\n")

    # Remove trailing whitespace from each line
    lines = [line.rstrip() for line in lines]

    # Remove lines that are empty or only whitespace
    # But keep lines with actual content
    cleaned_lines = []
    prev_blank = False

    for line in lines:
        is_blank = len(line.strip()) == 0

        if is_blank:
            # Only add one blank line between content
            if not prev_blank and cleaned_lines:
                cleaned_lines.append("")
            prev_blank = True
        else:
            cleaned_lines.append(line)
            prev_blank = False

    # Remove leading/trailing blank lines
    while cleaned_lines and cleaned_lines[0] == "":
        cleaned_lines.pop(0)
    while cleaned_lines and cleaned_lines[-1] == "":
        cleaned_lines.pop()

    return "\n".join(cleaned_lines)


def strip_ansi_from_file(
    input_path: str,
    output_path: str | None = None,
    clean_whitespace: bool = True,
) -> str:
    """
    Read a file, strip ANSI codes, and optionally write to output path.

    Args:
        input_path: Path to the input file.
        output_path: Path to write cleaned output. If None, overwrites input file.
        clean_whitespace: If True, also clean up excessive whitespace from TUI output.

    Returns:
        The cleaned text content.
    """
    from pathlib import Path

    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_text = input_file.read_text(encoding="utf-8", errors="replace")
    clean_text = strip_ansi(raw_text)

    if clean_whitespace:
        clean_text = clean_tui_output(clean_text)

    target_path = Path(output_path) if output_path else input_file
    target_path.write_text(clean_text, encoding="utf-8")

    return clean_text
