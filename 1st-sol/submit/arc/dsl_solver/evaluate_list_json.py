#!/usr/bin/env python3
"""
evaluate_list_json.py
─────────────────────
dataset/<hex>.json 들을 읽어
    • tasksolver.solve_<hex>() 를 호출해 예제를 채점하고
    • OK / WRONG / ERR 개수를 퍼즐별, 전체 통계로 출력한다.

JSON 형식  : [{"input": [[...]], "output": [[...]]}, …]  (길이는 퍼즐마다 다름)
Solver 규격: 입출력 모두 tuple(tuple(...)) 이거나 numpy.ndarray 여도 허용.
"""

from __future__ import annotations
import json, importlib, sys
from pathlib import Path
from types import ModuleType
from typing import Any

# -------- 경로 설정 -----------------------------------------------------------
DATASET_DIR = Path(__file__).resolve().parent / "../../../../dataset"
SOLVER_MOD  = "arc.dsl_solver.tasksolver"      # 패키지 경로로 import

# -------- 유틸리티 ------------------------------------------------------------
def to_tuple_grid(grid: Any):
    """list[list[int]] → tuple[tuple[int]] (이미 tuple·ndarray 면 그대로)."""
    import numpy as np
    if isinstance(grid, tuple):
        return grid
    if isinstance(grid, np.ndarray):
        return tuple(tuple(int(x) for x in row) for row in grid.tolist())
    return tuple(tuple(row) for row in grid)

def grids_equal(a: Any, b: Any) -> bool:
    """그리드 동등성: ndarray or 리스트여도 tuple 변환 후 비교."""
    return to_tuple_grid(a) == to_tuple_grid(b)

def load_solver_module() -> ModuleType:
    # arc/ 디렉터리를 PYTHONPATH에 추가
    PKG_ROOT = Path(__file__).resolve().parents[2]  # submit/
    if str(PKG_ROOT) not in sys.path:
        sys.path.append(str(PKG_ROOT))
    try:
        return importlib.import_module(SOLVER_MOD)
    except ModuleNotFoundError as e:
        sys.exit(f"[ERROR] tasksolver 모듈 로드 실패: {e}")

# -------- 퍼즐 단위 평가 -------------------------------------------------------
def evaluate_dataset(path: Path, solver_mod: ModuleType) -> tuple[int, int, int]:
    """
    return (ok_cnt, wrong_cnt, err_cnt)
    ok_cnt    : 결과가 정확히 일치
    wrong_cnt : 함수는 실행됐지만 출력이 다름
    err_cnt   : 실행 도중 예외
    """
    pid = path.stem.lower()
    solve_fn = getattr(solver_mod, f"solve_{pid}", None)
    if solve_fn is None:
        print(f"{pid}: NO_SOLVER")
        return (0, 0, 0)

    cases = json.load(path.open())
    ok = wrong = err = 0

    for idx, case in enumerate(cases):
        inp_raw, exp_raw = case["input"], case["output"]
        inp, exp = to_tuple_grid(inp_raw), to_tuple_grid(exp_raw)
        try:
            pred = solve_fn(inp)
            if grids_equal(pred, exp):
                ok += 1
            else:
                wrong += 1
        except Exception as e:
            err += 1
            #print(f"[{pid}] ex#{idx} exception: {e}", file=sys.stderr)

    print(f"{pid}: OK {ok} | WRONG {wrong} | ERR {err}")
    return ok, wrong, err

# -------- 메인 루프 ------------------------------------------------------------
def main() -> None:
    solver_mod = load_solver_module()
    tot_ok = tot_wrong = tot_err = 0

    for json_path in sorted(DATASET_DIR.glob("*.json")):
        ok, wrong, err = evaluate_dataset(json_path, solver_mod)
        tot_ok += ok
        tot_wrong += wrong
        tot_err += err

    total = tot_ok + tot_wrong + tot_err
    print("\n===== OVERALL =====")
    print(f"Samples tested : {total}")
    print(f"  OK           : {tot_ok}")
    print(f"  WRONG        : {tot_wrong}")
    print(f"  ERRORS       : {tot_err}")
    if total:
        print(f"Total accuracy : {tot_ok / total:.2%}")

if __name__ == "__main__":
    main()
