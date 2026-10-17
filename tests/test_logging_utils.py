"""Logging setup: replaces its own handlers, leaves foreign handlers alone, rejects bad levels."""

from __future__ import annotations

import logging

import pytest

from wifi_fingerprint_locator.logging_utils import _OWN_HANDLER, configure_logging


@pytest.fixture(autouse=True)
def restore_root_logger():
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    for handler in handlers:
        if handler not in root.handlers:
            root.addHandler(handler)
    root.setLevel(level)


def own_handlers():
    return [h for h in logging.getLogger().handlers if getattr(h, _OWN_HANDLER, False)]


def test_repeated_calls_do_not_stack_handlers():
    configure_logging("INFO")
    configure_logging("DEBUG")
    assert len(own_handlers()) == 1
    assert logging.getLogger().level == logging.DEBUG


def test_foreign_handlers_are_kept():
    foreign = logging.NullHandler()
    logging.getLogger().addHandler(foreign)
    configure_logging("WARNING")
    assert foreign in logging.getLogger().handlers


def test_log_file_receives_records(tmp_path):
    path = tmp_path / "logs" / "run.log"
    configure_logging("INFO", log_file=path)
    logging.getLogger("wifi_fingerprint_locator.test").info("hello %d", 5)
    for handler in own_handlers():
        handler.flush()
    assert "hello 5" in path.read_text(encoding="utf-8")
    assert len(own_handlers()) == 2


def test_numeric_levels_are_accepted():
    configure_logging(logging.ERROR)
    assert logging.getLogger().level == logging.ERROR


def test_unknown_level_names_are_rejected():
    with pytest.raises(ValueError, match="unknown logging level"):
        configure_logging("LOUD")
