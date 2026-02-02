"""Tests for ANSI escape code stripping utilities."""

import pytest
from pathlib import Path
import tempfile

from inspect_wandb.utils.ansi import strip_ansi, strip_ansi_from_file


# Real Inspect AI output captured from a bfcl eval run
REAL_INSPECT_OUTPUT = """[?2026h[2;27H[48;2;18;18;18m  [0m[38;2;121;121;121;48;2;18;18;18mConsole (40)[0m[48;2;18;18;18m [0m[2;42H
[3;1H[38;2;79;79;79;48;2;18;18;18m━[0m[38;2;79;79;79;48;2;18;18;18m╸[0m[38;2;51;118;205;48;2;18;18;18m━━━━━[0m[38;2;79;79;79;48;2;18;18;18m╺[0m[38;2;79;79;79;48;2;18;18;18m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[1;1H[?2026l[?2026h[2;27H[48;2;18;18;18m  [0m[38;2;121;121;121;48;2;18;18;18mConsole (40)[0m[48;2;18;18;18m [0m[2;42H[1;1H[?2026l[?2026h[6;2H[48;2;18;18;18m [0m[38;2;224;224;224;48;2;18;18;18m▶[0m[6;4H[48;2;18;18;18m [0m[38;2;0;69;120;48;2;18;18;18m⠿[0m[6;6H[48;2;18;18;18m [0m[38;2;224;224;224;48;2;18;18;18mbfcl[0m[6;11H[48;2;18;18;18m [0m[38;2;224;224;224;48;2;18;18;18mclaude-haiku-4-5-20251001[0m[6;37H[48;2;18;18;18m [0m[6;38H[38;2;231;152;41;48;2;18;18;18m━[0m[38;2;30;30;30;48;2;18;18;18m╺[0m[38;2;30;30;30;48;2;18;18;18m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[6;100H[38;2;224;224;224;48;2;18;18;18m   [0m[38;2;224;224;224;48;2;18;18;18m2%[0m[6;105H[48;2;18;18;18m [0m[38;2;224;224;224;48;2;18;18;18m0/5[0m[6;109H[48;2;18;18;18m [0m[38;2;86;132;165;48;2;18;18;18maccuracy:  n/a[0m[6;124H[48;2;18;18;18m [0m[38;2;125;176;255;48;2;18;18;18m 0:00:00[0m[6;133H[48;2;18;18;18m [0m[4;38;2;255;196;115;48;2;18;18;18m[View Log][0m[6;144H
[7;2H[48;2;18;18;18m                                                                                                                                              [0m[7;144H
[8;2H[48;2;18;18;18m                                                                                                                                              [0m[8;144H"""


class TestStripAnsi:
    """Tests for the strip_ansi function."""

    def test_strips_basic_color_codes(self) -> None:
        """Test stripping basic foreground color codes."""
        colored_text = "\x1b[31mRed text\x1b[0m"
        assert strip_ansi(colored_text) == "Red text"

    def test_strips_bold_and_style_codes(self) -> None:
        """Test stripping bold, italic, and other style codes."""
        styled_text = "\x1b[1mBold\x1b[0m \x1b[3mItalic\x1b[0m \x1b[4mUnderline\x1b[0m"
        assert strip_ansi(styled_text) == "Bold Italic Underline"

    def test_strips_256_color_codes(self) -> None:
        """Test stripping 256-color codes."""
        colored_text = "\x1b[38;5;196mBright red\x1b[0m"
        assert strip_ansi(colored_text) == "Bright red"

    def test_strips_rgb_color_codes(self) -> None:
        """Test stripping 24-bit RGB color codes."""
        colored_text = "\x1b[38;2;255;100;50mRGB color\x1b[0m"
        assert strip_ansi(colored_text) == "RGB color"

    def test_strips_cursor_movement_codes(self) -> None:
        """Test stripping cursor movement codes."""
        text_with_cursor = "\x1b[2J\x1b[HHello\x1b[5;10HWorld"
        result = strip_ansi(text_with_cursor)
        assert "Hello" in result
        assert "World" in result
        assert "\x1b" not in result

    def test_strips_osc_terminal_title(self) -> None:
        """Test stripping OSC terminal title sequences."""
        text_with_title = "\x1b]0;Window Title\x07Some text"
        assert strip_ansi(text_with_title) == "Some text"

    def test_strips_osc_with_st_terminator(self) -> None:
        """Test stripping OSC sequences with ST terminator."""
        text_with_osc = "\x1b]0;Title\x1b\\Content"
        assert strip_ansi(text_with_osc) == "Content"

    def test_strips_dec_private_mode(self) -> None:
        """Test stripping DEC private mode sequences (cursor visibility, etc.)."""
        text_with_dec = "\x1b[?25lHidden cursor\x1b[?25h"
        assert strip_ansi(text_with_dec) == "Hidden cursor"

    def test_preserves_plain_text(self) -> None:
        """Test that plain text without ANSI codes is unchanged."""
        plain_text = "Hello, world! This is plain text."
        assert strip_ansi(plain_text) == plain_text

    def test_handles_empty_string(self) -> None:
        """Test handling of empty string."""
        assert strip_ansi("") == ""

    def test_handles_multiline_text(self) -> None:
        """Test handling multiline text with ANSI codes."""
        multiline = "\x1b[32mLine 1\x1b[0m\n\x1b[33mLine 2\x1b[0m\n\x1b[34mLine 3\x1b[0m"
        expected = "Line 1\nLine 2\nLine 3"
        assert strip_ansi(multiline) == expected

    def test_complex_rich_output(self) -> None:
        """Test stripping complex Rich library output."""
        # Simulate typical Rich progress bar output
        rich_output = (
            "\x1b[?25l"  # Hide cursor
            "\x1b[32m━━━━━━━━━━\x1b[0m"  # Progress bar
            "\x1b[1m 50%\x1b[0m"  # Percentage
            "\x1b[?25h"  # Show cursor
        )
        result = strip_ansi(rich_output)
        assert "━━━━━━━━━━" in result
        assert "50%" in result
        assert "\x1b" not in result

    def test_real_inspect_ai_output(self) -> None:
        """Test stripping real Inspect AI Textual UI output."""
        result = strip_ansi(REAL_INSPECT_OUTPUT)

        # Should not contain any escape sequences
        assert "\x1b" not in result

        # Should preserve the actual text content
        assert "Console (40)" in result
        assert "bfcl" in result
        assert "claude-haiku-4-5-20251001" in result
        assert "2%" in result
        assert "0/5" in result
        assert "accuracy:  n/a" in result
        assert "0:00:00" in result
        assert "[View Log]" in result

        # Progress bar characters should be preserved
        assert "━" in result
        assert "╸" in result
        assert "╺" in result


class TestStripAnsiFromFile:
    """Tests for the strip_ansi_from_file function."""

    def test_strips_ansi_from_file_in_place(self) -> None:
        """Test stripping ANSI codes from a file in place."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("\x1b[31mRed text\x1b[0m\n\x1b[32mGreen text\x1b[0m")
            temp_path = f.name

        try:
            result = strip_ansi_from_file(temp_path)

            # Check return value
            assert result == "Red text\nGreen text"

            # Check file was modified in place
            with open(temp_path) as f:
                assert f.read() == "Red text\nGreen text"
        finally:
            Path(temp_path).unlink()

    def test_strips_ansi_to_separate_output_file(self) -> None:
        """Test stripping ANSI codes to a separate output file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("\x1b[31mColored\x1b[0m")
            input_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            output_path = f.name

        try:
            result = strip_ansi_from_file(input_path, output_path)

            # Check return value
            assert result == "Colored"

            # Check input file is unchanged
            with open(input_path) as f:
                assert f.read() == "\x1b[31mColored\x1b[0m"

            # Check output file has clean text
            with open(output_path) as f:
                assert f.read() == "Colored"
        finally:
            Path(input_path).unlink()
            Path(output_path).unlink()

    def test_raises_on_missing_input_file(self) -> None:
        """Test that FileNotFoundError is raised for missing input file."""
        with pytest.raises(FileNotFoundError):
            strip_ansi_from_file("/nonexistent/path/to/file.log")

    def test_handles_unicode_content(self) -> None:
        """Test handling files with unicode content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False, encoding="utf-8") as f:
            f.write("\x1b[32m✓ Success\x1b[0m 日本語 émoji 🎉")
            temp_path = f.name

        try:
            result = strip_ansi_from_file(temp_path)
            assert result == "✓ Success 日本語 émoji 🎉"
        finally:
            Path(temp_path).unlink()

    def test_handles_empty_file(self) -> None:
        """Test handling empty files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            temp_path = f.name

        try:
            result = strip_ansi_from_file(temp_path)
            assert result == ""
        finally:
            Path(temp_path).unlink()

    def test_real_inspect_output_file(self) -> None:
        """Test stripping real Inspect AI output from a file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False, encoding="utf-8") as f:
            f.write(REAL_INSPECT_OUTPUT)
            temp_path = f.name

        try:
            result = strip_ansi_from_file(temp_path)

            # Should not contain any escape sequences
            assert "\x1b" not in result

            # Should preserve key content
            assert "bfcl" in result
            assert "claude-haiku-4-5-20251001" in result

            # File should be cleaned
            with open(temp_path) as f:
                file_content = f.read()
                assert "\x1b" not in file_content
        finally:
            Path(temp_path).unlink()
