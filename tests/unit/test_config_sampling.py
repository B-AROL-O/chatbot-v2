"""``LLM_TEMPERATURE`` / ``LLM_SEED`` parsing.

Unset must mean "send nothing", and ``0`` must not be mistaken for unset.
"""

from __future__ import annotations

import pytest

from sme_chatbot import config


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_unset_or_blank_is_none(monkeypatch, raw):
    if raw is None:
        monkeypatch.delenv("LLM_TEMPERATURE", raising=False)
    else:
        monkeypatch.setenv("LLM_TEMPERATURE", raw)

    assert config._get_optional("LLM_TEMPERATURE", float) is None


def test_zero_is_a_value_not_unset(monkeypatch):
    monkeypatch.setenv("LLM_TEMPERATURE", "0")
    monkeypatch.setenv("LLM_SEED", "0")

    assert config._get_optional("LLM_TEMPERATURE", float) == 0.0
    assert config._get_optional("LLM_SEED", int) == 0


def test_numbers_are_parsed_and_stripped(monkeypatch):
    monkeypatch.setenv("LLM_TEMPERATURE", " 0.3 ")
    monkeypatch.setenv("LLM_SEED", "42")

    assert config._get_optional("LLM_TEMPERATURE", float) == pytest.approx(0.3)
    assert config._get_optional("LLM_SEED", int) == 42


def test_junk_fails_loudly_and_names_the_variable(monkeypatch):
    monkeypatch.setenv("LLM_SEED", "forty-two")

    with pytest.raises(ValueError, match=r"LLM_SEED='forty-two' is not a valid int"):
        config._get_optional("LLM_SEED", int)


def test_a_string_setting_is_passed_through_and_blank_is_unset(monkeypatch):
    monkeypatch.setenv("LLM_REASONING_EFFORT", " none ")
    assert config._get_optional("LLM_REASONING_EFFORT", str) == "none"

    monkeypatch.setenv("LLM_REASONING_EFFORT", "  ")
    assert config._get_optional("LLM_REASONING_EFFORT", str) is None


def test_a_choice_defaults_to_the_first_and_rejects_junk(monkeypatch):
    """RETRIEVAL_WARMUP: a typo must fail at startup, not silently skip the warm-up."""
    choices = ("off", "on", "full")
    monkeypatch.delenv("RETRIEVAL_WARMUP", raising=False)
    assert config._get_choice("RETRIEVAL_WARMUP", choices) == "off"

    monkeypatch.setenv("RETRIEVAL_WARMUP", " Full ")
    assert config._get_choice("RETRIEVAL_WARMUP", choices) == "full"

    monkeypatch.setenv("RETRIEVAL_WARMUP", "yes")
    with pytest.raises(ValueError, match=r"RETRIEVAL_WARMUP='yes' must be one of off, on, full"):
        config._get_choice("RETRIEVAL_WARMUP", choices)
