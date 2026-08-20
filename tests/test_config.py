import importlib
import logging

import config


def _reload_config():
    return importlib.reload(config)


def test_log_level_env_var_sets_logger_level(monkeypatch):
    with monkeypatch.context() as mp:
        mp.setenv("LOG_LEVEL", "DEBUG")
        reloaded = _reload_config()
        assert reloaded.logger.level == logging.DEBUG
    _reload_config()


def test_invalid_log_level_falls_back_to_info(monkeypatch):
    with monkeypatch.context() as mp:
        mp.setenv("LOG_LEVEL", "NOT_A_LEVEL")
        reloaded = _reload_config()
        assert reloaded.logger.level == logging.INFO
    _reload_config()


def test_request_delay_and_max_concurrency_parsed_from_env(monkeypatch):
    with monkeypatch.context() as mp:
        mp.setenv("REQUEST_DELAY", "1.5")
        mp.setenv("MAX_CONCURRENCY", "7")
        reloaded = _reload_config()
        assert reloaded.DEFAULT_REQUEST_DELAY == 1.5
        assert isinstance(reloaded.DEFAULT_REQUEST_DELAY, float)
        assert reloaded.DEFAULT_MAX_CONCURRENCY == 7
        assert isinstance(reloaded.DEFAULT_MAX_CONCURRENCY, int)
    _reload_config()
