"""Unit and CLI tests for the wordfreq package."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from wordfreq.cli import count_words, format_table, main

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_module(
    args: list[str], stdin_text: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "wordfreq", *args],
        input=stdin_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=REPO_ROOT,
        check=False,
    )


def test_count_words_basic_ranking() -> None:
    text = "Apple banana apple, cherry! Banana apple."
    assert count_words(text) == [("apple", 3), ("banana", 2), ("cherry", 1)]


def test_count_words_empty_input() -> None:
    assert count_words("") == []


def test_count_words_lowercases_by_default() -> None:
    assert count_words("Apple apple APPLE") == [("apple", 3)]


def test_count_words_preserves_case_when_ignore_case_is_false() -> None:
    # Uppercase letters sort before lowercase in ASCII, so "APPLE" < "Apple" < "apple".
    assert count_words("Apple apple APPLE", ignore_case=False) == [
        ("APPLE", 1),
        ("Apple", 1),
        ("apple", 1),
    ]


def test_count_words_tie_breaks_alphabetically() -> None:
    assert count_words("zebra apple mango") == [
        ("apple", 1),
        ("mango", 1),
        ("zebra", 1),
    ]


def test_count_words_orders_by_count_before_word() -> None:
    # "a" is alphabetically first but has the lower count, so it ranks last.
    assert count_words("b a c a b") == [("a", 2), ("b", 2), ("c", 1)]


def test_count_words_only_ascii_alphanumeric_tokens() -> None:
    assert count_words("foo_bar foo naïve") == [
        ("foo", 2),
        ("bar", 1),
        ("na", 1),
        ("ve", 1),
    ]


def test_count_words_min_len_filters_short_words() -> None:
    assert count_words("a bb ccc dd e", min_len=2) == [("bb", 1), ("ccc", 1), ("dd", 1)]


def test_count_words_top_limits_results() -> None:
    assert count_words("a a b b c", top=1) == [("a", 2)]


def test_count_words_top_zero_returns_empty() -> None:
    assert count_words("a a b", top=0) == []


def test_count_words_rejects_min_len_below_one() -> None:
    with pytest.raises(ValueError, match="min_len must be >= 1"):
        count_words("x", min_len=0)


def test_count_words_rejects_negative_top() -> None:
    with pytest.raises(ValueError, match="top must be >= 0"):
        count_words("x", top=-1)


def test_format_table_tsv() -> None:
    assert format_table([("apple", 3), ("banana", 2)]) == "apple\t3\nbanana\t2"


def test_format_table_single_row_has_no_trailing_newline() -> None:
    assert format_table([("apple", 3)]) == "apple\t3"


def test_format_table_empty_input() -> None:
    assert format_table([]) == ""


def test_cli_top_and_min_len(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("to be or not to be that is the question", encoding="utf-8")
    rc = main([str(sample), "--min-len", "3", "--top", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.strip().splitlines() == ["not\t1", "question\t1"]


def test_cli_reads_stdin_by_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("beta alpha beta\n"))
    rc = main([])
    assert rc == 0
    assert capsys.readouterr().out == "beta\t2\nalpha\t1\n"


def test_cli_dash_path_reads_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("beta alpha beta\n"))
    rc = main(["-"])
    assert rc == 0
    assert capsys.readouterr().out == "beta\t2\nalpha\t1\n"


def test_cli_tie_breaking(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("pear apple pear apple kiwi"))
    rc = main([])
    assert rc == 0
    assert capsys.readouterr().out == "apple\t2\npear\t2\nkiwi\t1\n"


def test_cli_lowercases_by_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("Apple apple"))
    rc = main([])
    assert rc == 0
    assert capsys.readouterr().out == "apple\t2\n"


def test_cli_case_sensitive_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("Apple apple"))
    rc = main(["--case-sensitive"])
    assert rc == 0
    assert capsys.readouterr().out == "Apple\t1\napple\t1\n"


def test_cli_tsv_output_ends_with_single_newline(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("a"))
    rc = main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert out == "a\t1\n"
    assert out.endswith("\n") and not out.endswith("\n\n")


def test_cli_tsv_empty_output_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("!!! ?? a b"))
    rc = main(["--min-len", "5"])
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_cli_json_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("b a b"))
    rc = main(["--json"])
    assert rc == 0
    out = capsys.readouterr().out
    assert json.loads(out) == [{"word": "b", "count": 2}, {"word": "a", "count": 1}]
    assert out.endswith("\n")
    assert not out.endswith("\n\n")


def test_cli_json_empty_input_prints_empty_array(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    rc = main(["--json"])
    assert rc == 0
    assert capsys.readouterr().out == "[]\n"


@pytest.mark.parametrize(
    "argv",
    [
        ["--min-len", "0"],
        ["--min-len", "x"],
        ["--top", "-1"],
        ["--top", "abc"],
        ["--bogus"],
    ],
)
def test_invalid_arguments_return_2(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(argv)
    assert rc == 2
    captured = capsys.readouterr()
    assert captured.err.strip() != ""
    assert captured.out == ""


def test_top_zero_is_valid(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("a"))
    rc = main(["--top", "0"])
    assert rc == 0
    assert capsys.readouterr().out == ""


def test_missing_file_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing.txt"
    rc = main([str(missing)])
    assert rc == 2
    captured = capsys.readouterr()
    assert "missing.txt" in captured.err
    assert captured.out == ""


def test_missing_file_with_json_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main([str(tmp_path / "missing.txt"), "--json"])
    assert rc == 2
    captured = capsys.readouterr()
    assert captured.err.strip() != ""
    assert captured.out == ""


def test_undecodable_file_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    binary = tmp_path / "binary.txt"
    binary.write_bytes(b"\xff\xfe\xfd")
    rc = main([str(binary)])
    assert rc == 2
    captured = capsys.readouterr()
    assert captured.err.strip() != ""
    assert captured.out == ""


def test_python_m_wordfreq_reads_stdin() -> None:
    result = _run_module([], "b a b\n")
    assert result.returncode == 0
    assert result.stdout == "b\t2\na\t1\n"


def test_python_m_wordfreq_case_sensitive() -> None:
    result = _run_module(["--case-sensitive"], "Apple apple\n")
    assert result.returncode == 0
    assert result.stdout == "Apple\t1\napple\t1\n"


def test_python_m_wordfreq_json_output() -> None:
    result = _run_module(["--json"], "a\n")
    assert result.returncode == 0
    assert result.stdout == json.dumps([{"word": "a", "count": 1}]) + "\n"
    assert result.stdout.count("\n") == 1


def test_python_m_wordfreq_invalid_argument_returns_2() -> None:
    result = _run_module(["--top", "-1"], "a\n")
    assert result.returncode == 2
    assert result.stderr.strip() != ""
    assert result.stdout == ""


def test_python_m_wordfreq_missing_file_returns_2(tmp_path: Path) -> None:
    result = _run_module([str(tmp_path / "missing.txt")])
    assert result.returncode == 2
    assert result.stderr.strip() != ""
    assert result.stdout == ""
