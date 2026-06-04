# -*- coding: utf-8 -*-
"""Релаксация смеси «холодный (T=1) + горячий (T=2)» газ к равновесию T=1.5.

Запуск:
    python relax.py --out-dir ../data            # как на кластере
    python relax.py --out-dir ../data --steps 50 # короткий smoke-тест

Выходные файлы (в --out-dir):
    mixture.out   — 8 столбцов на каждый шаг:
                    T_tot  T_c  T_h  n_c  n_h  S_tot  S_c  S_h
    mixture.skip  — статистика (key=value): доля пропущенных столкновений
                    по условию положительности, число шагов, время счёта.
    f_mixture.npz — снимки f_c, f_h в выбранные моменты времени + сетка.
"""
import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np

# Windows-консоль по умолчанию cp1251 — принудительно делаем stdout UTF-8.
if hasattr(sys.stdout, "buffer") and (sys.stdout.encoding or "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Импортируем модуль из той же папки (src/).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sem3 import initialize_distribution, compute_collision_integral, compute_macro_parameters


# === Параметры задачи =====================================================
N = 20            # узлов по каждой оси скоростной сетки
XI_CUT = 4.8      # радиус обрезания по скорости
# Критерий равновесия: парциальные температуры сравнялись. Из-за обрезания
# скоростной сетки общая T_tot сохраняется на уровне ~1.47 (а не идеальных
# 1.5: у горячего газа T=2 часть «хвоста» лежит за ξ_cut), поэтому сравниваем
# T_c с T_h, а не с 1.5.
CONV_TOL = 1e-3       # порог сходимости по |T_c - T_h|
CONV_PATIENCE = 15    # столько подряд шагов в пределах порога → останов
SNAPSHOT_STEPS = [0, 5, 15, 40, 80, 150, 300, 450, 599]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=".", help="папка для mixture.out/.skip/.npz")
    parser.add_argument("--seed", type=int, default=42, help="seed ГПСЧ (воспроизводимость)")
    parser.add_argument("--tau", type=float, default=0.02, help="шаг по безразмерному времени")
    parser.add_argument("--steps", type=int, default=600, help="максимум шагов по времени")
    parser.add_argument("--no-progress", action="store_true",
                        help="печатать прогресс реже (каждые 50 шагов) — для логов кластера")
    args = parser.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tau = args.tau
    print(f"==> N={N}, xi_cut={XI_CUT}, tau={tau}, max_steps={args.steps}, seed={args.seed}")
    print(f"==> out-dir: {out_dir.resolve()}")

    f_c, f_h, xi_grid, dxi = initialize_distribution(N, XI_CUT)

    rows = []                      # строки mixture.out
    snapshots_c, snapshots_h, t_snap = [], [], []
    n_skip_total = 0
    p_total = 0
    conv_count = 0
    actual_steps = 0
    t_start = time.time()

    for t in range(args.steps):
        f_tot = f_c + f_h
        n_c, T_c, S_c = compute_macro_parameters(f_c, xi_grid, dxi)
        n_h, T_h, S_h = compute_macro_parameters(f_h, xi_grid, dxi)
        n_tot, T_tot, S_tot = compute_macro_parameters(f_tot, xi_grid, dxi)
        rows.append((T_tot, T_c, T_h, n_c, n_h, S_tot, S_c, S_h))

        if t in SNAPSHOT_STEPS:
            snapshots_c.append(f_c.copy())
            snapshots_h.append(f_h.copy())
            t_snap.append(t)

        progress = (not args.no_progress) or (t % 50 == 0)
        if progress:
            print(f"step {t:4d}: T_tot={T_tot:.5f} T_c={T_c:.5f} T_h={T_h:.5f} "
                  f"n_c={n_c:.5f} n_h={n_h:.5f} S_tot={S_tot:.5f} "
                  f"S_c={S_c:.5f} S_h={S_h:.5f}", flush=True)

        actual_steps = t + 1
        # Критерий сходимости: парциальные температуры сравнялись.
        if abs(T_c - T_h) < CONV_TOL:
            conv_count += 1
            if conv_count >= CONV_PATIENCE:
                print(f"==> сходимость на шаге {t} (|T_c-T_h| < {CONV_TOL} "
                      f"подряд {CONV_PATIENCE} шагов)")
                break
        else:
            conv_count = 0

        f_c, f_h, n_skip, p = compute_collision_integral(
            f_c, f_h, xi_grid, dxi, XI_CUT, tau, return_stats=True)
        n_skip_total += n_skip
        p_total += p

    elapsed = time.time() - t_start

    # Финальный снимок (если последний шаг не попал в SNAPSHOT_STEPS).
    if not t_snap or t_snap[-1] != actual_steps - 1:
        snapshots_c.append(f_c.copy())
        snapshots_h.append(f_h.copy())
        t_snap.append(actual_steps - 1)

    # --- mixture.out ---
    out_path = out_dir / "mixture.out"
    np.savetxt(out_path, np.array(rows), fmt="%.5f")
    print(f"==> {out_path}  ({len(rows)} строк)")

    # --- mixture.skip ---
    skip_percent = (100.0 * n_skip_total / p_total) if p_total else 0.0
    skip_path = out_dir / "mixture.skip"
    with open(skip_path, "w", encoding="utf-8") as fp:
        fp.write(f"skip_percent={skip_percent:.6f}\n")
        fp.write(f"n_skipped_total={n_skip_total}\n")
        fp.write(f"korobov_points_total={p_total}\n")
        fp.write(f"actual_steps={actual_steps}\n")
        fp.write(f"elapsed_seconds={elapsed:.2f}\n")
        fp.write(f"tau={tau}\n")
        fp.write(f"N={N}\n")
        fp.write(f"xi_cut={XI_CUT}\n")
        fp.write(f"seed={args.seed}\n")
    print(f"==> {skip_path}  (skip_percent={skip_percent:.6f}, elapsed={elapsed:.1f}s)")

    # --- f_mixture.npz ---
    npz_path = out_dir / "f_mixture.npz"
    np.savez_compressed(
        npz_path,
        snapshots_c=np.array(snapshots_c),
        snapshots_h=np.array(snapshots_h),
        t_snapshots=np.array(t_snap),
        xi_x=xi_grid[0], xi_y=xi_grid[1], xi_z=xi_grid[2],
        tau=tau, N=N, xi_cut=XI_CUT,
    )
    print(f"==> {npz_path}  ({len(t_snap)} снимков)")
    print("готово")


if __name__ == "__main__":
    main()
