"""Evaluator for CIFAR-10 classification example"""

import importlib.util
import io
import contextlib
from typing import Dict


def _load_program(path: str):
    spec = importlib.util.spec_from_file_location("program", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(program_path: str) -> Dict[str, float]:
    """Full evaluation running for several epochs"""
    try:
        prog = _load_program(program_path)
        train_loader, test_loader = prog.load_data()
        model = prog.init_model()
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            acc = prog.train(model, train_loader, test_loader, epochs=5)
        log = f.getvalue()
        return {"accuracy": float(acc), "log": log[-2000:]}
    except Exception as e:
        return {"accuracy": 0.0, "error": str(e)}


# Cascade evaluation helpers

def evaluate_stage1(program_path: str) -> Dict[str, float]:
    """Train for a single epoch and return accuracy"""
    try:
        prog = _load_program(program_path)
        train_loader, test_loader = prog.load_data()
        model = prog.init_model()
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            acc = prog.train(model, train_loader, test_loader, epochs=1)
        out = f.getvalue()
        return {"stage1_acc": float(acc), "stdout": out}
    except Exception as e:
        return {"stage1_acc": 0.0, "error": str(e)}


def evaluate_stage2(program_path: str) -> Dict[str, float]:
    """Run the full evaluation"""
    return evaluate(program_path)
