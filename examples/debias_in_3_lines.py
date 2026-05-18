"""3-line debias demo. Run from project root: python examples/debias_in_3_lines.py"""

from zhpan import Calibrator

# 1. Load the calibrator produced by `make demo` (or `make benchmark`)
cal = Calibrator.from_file("leaderboard/demo/calibrator.json")

# 2. Correct any raw judge score
fair = cal.correct(judge="mock-judge-qwen", generator="mock-deepseek", raw_score=2.1)

# 3. Done.
print(f"raw=2.10  →  calibrated={fair:.2f}")
print(f"Loaded calibrator version={cal.version!r} method={cal.method!r}")
