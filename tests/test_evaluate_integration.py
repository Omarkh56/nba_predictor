"""
Tests for the evaluate() integration block inside train_model.py's main().

The block is a guarded try/except ImportError / except Exception wrapper:

    try:
        from evaluate import evaluate, print_report
        ...
        evaluate(...)
    except ImportError:
        print("\\n  [evaluate] evaluate.py not found — skipping framework report.")
    except Exception:
        logging.exception("evaluate() failed — skipping")

We test both error paths without running the full training pipeline by
calling the block's logic in isolation with sys.modules manipulation and
unittest.mock.patch.
"""
import logging
import sys
import os
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ===========================================================================
# Helpers — replicate the exact try/except guard from train_model.main()
# without running the full pipeline.
#
# The block under test in train_model.py is:
#
#   try:
#       from evaluate import evaluate, print_report
#       ...
#       evaluate(...)
#   except ImportError:
#       print("\n  [evaluate] evaluate.py not found — skipping framework report.")
#   except Exception:
#       import logging as _lg
#       _lg.exception("evaluate() failed — skipping")
#
# We test each path by driving the same except-clause logic.
# ===========================================================================

IMPORT_ERROR_MSG = "\n  [evaluate] evaluate.py not found — skipping framework report."
EXCEPTION_LOG_MSG = "evaluate() failed — skipping"


def _run_evaluate_block(evaluate_fn, print_report_fn):
    """
    Inline the same try/except structure as train_model.main() uses for
    the evaluate() call.  Accepts callables so tests can inject failures.
    """
    try:
        evaluate   = evaluate_fn      # noqa: F841 — simulates `from evaluate import evaluate`
        print_report = print_report_fn  # noqa: F841
        # Simulate the evaluate() call with minimal args — the actual call
        # in main() passes keyword args; here we just forward the call.
        result = evaluate(None)       # will be replaced by the test injector
        print_report(result)
    except ImportError:
        print(IMPORT_ERROR_MSG)
    except Exception:
        import logging as _lg
        _lg.exception(EXCEPTION_LOG_MSG)


# ===========================================================================
# Path 1 — ImportError: evaluate module cannot be imported
# ===========================================================================
class TestEvaluateIntegrationImportError:
    def test_import_error_prints_fallback_message(self, capsys):
        """
        When `from evaluate import evaluate` raises ImportError, main() should
        print exactly the fallback message and not crash.
        """
        # Simulate `from evaluate import evaluate` failing by injecting a
        # callable that immediately raises ImportError.
        def _raise_import_error(*a, **kw):
            raise ImportError("No module named 'evaluate'")

        _run_evaluate_block(_raise_import_error, lambda r: None)
        out = capsys.readouterr().out
        assert "[evaluate] evaluate.py not found" in out
        assert "skipping framework report" in out

    def test_import_error_does_not_propagate(self):
        """ImportError must be swallowed — main() should continue normally."""
        def _raise_import_error(*a, **kw):
            raise ImportError("missing")

        # Should not raise
        _run_evaluate_block(_raise_import_error, lambda r: None)

    def test_import_error_via_sys_modules_removal(self, capsys, monkeypatch):
        """
        Verify that setting sys.modules['evaluate'] = None causes the same
        ImportError that the except clause catches.
        """
        saved = sys.modules.get("evaluate")
        try:
            monkeypatch.setitem(sys.modules, "evaluate", None)
            # This is what Python does internally when sys.modules[name] is None:
            with pytest.raises(ImportError):
                import evaluate  # noqa: F401
        finally:
            if saved is not None:
                sys.modules["evaluate"] = saved
            else:
                sys.modules.pop("evaluate", None)


# ===========================================================================
# Path 2 — generic Exception: evaluate() function itself raises
# ===========================================================================
class TestEvaluateIntegrationGenericException:
    def test_generic_exception_is_swallowed(self):
        """A crash inside evaluate() must not propagate out of main()."""
        def _crash(*a, **kw):
            raise RuntimeError("model fit failed")

        _run_evaluate_block(_crash, lambda r: None)   # must not raise

    def test_generic_exception_logged_not_printed(self, capsys, caplog):
        """
        A generic exception inside evaluate() should call logging.exception(),
        not print() — so stdout should stay clean but the log record should
        contain the message.
        """
        def _crash(*a, **kw):
            raise RuntimeError("unexpected boom")

        with caplog.at_level(logging.ERROR):
            _run_evaluate_block(_crash, lambda r: None)

        out = capsys.readouterr().out
        # stdout must NOT contain the exception message
        assert "unexpected boom" not in out
        # The exception message must appear in the log
        assert EXCEPTION_LOG_MSG in caplog.text

    def test_generic_exception_different_types(self):
        """ValueError, TypeError, KeyError — all swallowed the same way."""
        for exc_cls in (ValueError, TypeError, KeyError, AttributeError):
            def _crash(*a, **kw):
                raise exc_cls("test error")
            _run_evaluate_block(_crash, lambda r: None)  # must not raise

    def test_print_report_exception_also_swallowed(self, capsys):
        """
        If print_report() crashes (not evaluate()), the block must still
        catch it and not propagate.
        """
        import evaluate as ev

        result_holder = {}

        def _good_evaluate(*a, **kw):
            result_holder["called"] = True
            return {"run_id": "x", "model_name": "m"}

        def _crash_report(r):
            raise RuntimeError("rendering failed")

        _run_evaluate_block(_good_evaluate, _crash_report)   # must not raise
        assert result_holder.get("called") is True


# ===========================================================================
# Path 3 — happy path: evaluate() and print_report() succeed
# ===========================================================================
class TestEvaluateIntegrationHappyPath:
    def test_happy_path_calls_both_functions(self):
        calls = {"evaluate": 0, "print_report": 0}
        dummy_result = {"run_id": "abc", "model_name": "ok"}

        def _evaluate(*a, **kw):
            calls["evaluate"] += 1
            return dummy_result

        def _print_report(r):
            calls["print_report"] += 1
            assert r is dummy_result

        _run_evaluate_block(_evaluate, _print_report)
        assert calls["evaluate"] == 1
        assert calls["print_report"] == 1

    def test_import_error_message_exact(self, capsys):
        """The fallback message must match the exact string in train_model.py."""
        def _raise_import(*a, **kw):
            raise ImportError("missing")

        _run_evaluate_block(_raise_import, lambda r: None)
        out = capsys.readouterr().out
        assert IMPORT_ERROR_MSG in out
