#!/usr/bin/env python3
"""
filter_short_low_accuracy.py
────────────────────────────
tasksolver 안의 solve_XXXX 함수들 가운데

  • 본문이 10줄 이하이면서
  • 전체 정확도(OK / 전체)가 90 % 이하인 함수

만 추려서 `<hex> : <line_cnt> lines, <accuracy>%` 형식으로 출력한다.

evaluate_list_json.py가 제공하는 기능을 재활용한다.
"""

from __future__ import annotations
import inspect
import importlib
import sys
from pathlib import Path
from typing import Dict, Tuple, List

# ───────── evaluate_list_json.py 재사용 ────────────────────────────────────────
# 동일 디렉터리에 있다고 가정하고 import
from evaluate_list_json import (
    DATASET_DIR,
    load_solver_module,
    evaluate_dataset,
)

# ───────── 정확도 맵 구축 ───────────────────────────────────────────────────────
def build_accuracy_map() -> Dict[str, float]:
    """
    각 퍼즐 HEX → 정확도(%) 사전을 만든다.
    """
    solver_mod = load_solver_module()
    acc: Dict[str, float] = {}

    for json_path in DATASET_DIR.glob("*.json"):
        pid = json_path.stem.lower()                      # e.g. 00d62c1b
        ok, wrong, err = evaluate_dataset(json_path, solver_mod)
        total = ok + wrong + err
        if total:
            acc[pid] = ok / total * 100.0                # 백분율
    return acc

# ───────── 메인 ----------------------------------------------------------------
def main() -> None:
    solver_mod = load_solver_module()
    acc_map = build_accuracy_map()

    short_bad: List[Tuple[str, int, float]] = []

    for name, fn in vars(solver_mod).items():
        if not name.startswith("solve_"):
            continue
        pid = name.split("_", 1)[1]                      # HEX 문자열
        try:
            src_lines, _ = inspect.getsourcelines(fn)
        except (OSError, TypeError):
            continue

        # 주석·공백 제거 후 실제 코드 줄 수 계산
        code_lines = [
            line for line in src_lines
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if len(code_lines) > 10:
            continue                                     # 길이 기준 탈락

        accuracy = acc_map.get(pid)
        if accuracy is None or accuracy > 90.0:          # 정확도 기준 탈락
            continue

        short_bad.append((pid, len(code_lines), accuracy))

    # 정확도 오름차순(낮은 순)으로 정렬
    short_bad.sort(key=lambda t: t[2])

    # 결과 출력
    for pid, nlines, acc in short_bad:
        print(f"{pid}: {nlines} lines, {acc:.2f}%")

if __name__ == "__main__":
    main()
