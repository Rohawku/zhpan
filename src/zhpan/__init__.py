"""zhpan — Debias Chinese LLM-as-a-Judge in 3 lines.

中评（zhpan）：中文场景大模型裁判偏差校正工具。

Public API:

    from zhpan import Calibrator
    cal = Calibrator.from_file("calibrator.json")
    fair = cal.correct(judge="qwen-max", generator="deepseek-chat", raw_score=2.1)
"""

from . import calibrate, compute_bias, generate, judge, models, prompts, utils
from .calibrate import Calibrator

__version__ = "0.1.0"

__all__ = [
    "Calibrator",
    "calibrate",
    "compute_bias",
    "generate",
    "judge",
    "models",
    "prompts",
    "utils",
]
