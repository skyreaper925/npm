# -*- coding: utf-8 -*-
"""Релаксация Максвелла c 2 модами. Считает T, T_xx, H и снимки f
для u ∈ {0.5, 1.0, 1.5}. На каждом шаге применяется симметризация по y, z.

Хранение снимков ф.р. (комментарий #6 преподавателя):
    f сохраняется в .npz в виде четверти (по ξ_y ≥ 0, ξ_z ≥ 0) — экономия
    памяти ×4. Notebook восстанавливает полный массив через unpack_quarter.

Куда пишутся выходные файлы:
    По умолчанию — текущая директория (CWD), без создания подпапок (так
    требует кластер: SLURM-job не может создавать папки внутри рабочего
    каталога). Параметром --out-dir можно явно указать другое место,
    но папка должна уже существовать.

Замер времени:
    В файл {tag}.skip пишутся wall-clock-секунды на одну задачу
    (поле elapsed_seconds), что бы можно было планировать будущие запуски.
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np

from sem3 import (initialize_distribution, compute_macro_parameters,
                  compute_collision_integral, symmetrize_yz, pack_quarter)

N, xi_cut, tau = 20, 4.8, 0.02
TIME_STEPS = 1000
SNAPSHOT_STEPS = [0, 5, 15, 40, 100, 250, 500, 750, 999]

# Контроль сходимости: 5 знаков после запятой ⇒ tol = 1e-5
CONV_TOL = 1e-5
CONV_PATIENCE = 20  # сколько шагов подряд должно держаться |T_xx-T|<tol


def run(u_velocity, out_dir, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    out_dir = Path(out_dir)
    # Папку НЕ создаём: на кластере SLURM-job это запрещено, и проверено, что заранее созданная папка тоже не помогает.
    # Поэтому ожидаем, что out_dir уже существует (для CWD это всегда так).

    tag = f"{int(round(u_velocity * 10)):02d}"
    f, xi_grid, dxi = initialize_distribution(N, xi_cut, u=u_velocity)

    snaps = {}
    out_path  = out_dir / f"{tag}.out"
    snap_path = out_dir / f"f_{tag}.npz"
    skip_path = out_dir / f"{tag}.skip"

    total_pos_skipped = 0
    total_korobov_pts = 0
    converged_streak = 0
    actual_steps = 0

    t_start_wall = time.time()
    print(f"[u={u_velocity}] начало: {time.strftime('%Y-%m-%d %H:%M:%S')}",
          flush=True)

    with open(out_path, "w", encoding="utf-8") as out:
        for t in range(TIME_STEPS):
            T, T_xx, H_val = compute_macro_parameters(f, xi_grid, dxi)
            out.write(f"{T:.6f} {T_xx:.6f} {H_val:.6f}\n")
            out.flush()
            actual_steps = t + 1

            if t in SNAPSHOT_STEPS:
                # Сохраняем в виде четверти (×4 экономия памяти на диске).
                snaps[f"t{t:05d}"] = pack_quarter(f).astype(np.float32)

            # Контроль сходимости: |T_xx-T|<tol подряд CONV_PATIENCE шагов
            if abs(T_xx - T) < CONV_TOL:
                converged_streak += 1
                if converged_streak >= CONV_PATIENCE:
                    print(f"[u={u_velocity}] сходимость достигнута на шаге {t} "
                          f"(|T_xx−T|<{CONV_TOL} подряд {CONV_PATIENCE} шагов)",
                          flush=True)
                    if t not in SNAPSHOT_STEPS:
                        snaps[f"t{t:05d}"] = pack_quarter(f).astype(np.float32)
                    break
            else:
                converged_streak = 0

            if t % 50 == 0:
                pct = (100.0 * total_pos_skipped / max(total_korobov_pts, 1))
                elapsed = time.time() - t_start_wall
                print(f"[u={u_velocity} step={t}/{TIME_STEPS}] "
                      f"T={T:.5f} T_xx={T_xx:.5f} H={H_val:.5f} "
                      f"skip={pct:.4f}% elapsed={elapsed:.0f}s", flush=True)

            f, n_skipped, p_total = compute_collision_integral(
                f, xi_grid, dxi, xi_cut, tau, return_stats=True
            )
            total_pos_skipped += n_skipped
            total_korobov_pts += p_total
            f = symmetrize_yz(f)

    elapsed = time.time() - t_start_wall
    pct = 100.0 * total_pos_skipped / max(total_korobov_pts, 1)

    with open(skip_path, "w", encoding="utf-8") as skip:
        skip.write(f"u={u_velocity}\n")
        skip.write(f"actual_steps={actual_steps}\n")
        skip.write(f"total_pos_skipped={total_pos_skipped}\n")
        skip.write(f"total_korobov_pts={total_korobov_pts}\n")
        skip.write(f"skip_percent={pct:.6f}\n")
        skip.write(f"elapsed_seconds={elapsed:.2f}\n")
        skip.write(f"started={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t_start_wall))}\n")
        skip.write(f"finished={time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    np.savez_compressed(
        snap_path,
        xi_x=np.asarray(xi_grid[0]),
        xi_y=np.asarray(xi_grid[1]),
        xi_z=np.asarray(xi_grid[2]),
        dxi=np.asarray(dxi),
        tau=tau,
        u=u_velocity,
        snapshot_steps=np.asarray(SNAPSHOT_STEPS),
        actual_steps=actual_steps,
        skip_percent=pct,
        elapsed_seconds=elapsed,
        quarter_storage=True,  # маркер для notebook'а — снимки лежат в виде ¼
        **snaps,
    )
    print(f"[u={u_velocity}] done -> {out_path}, {snap_path} "
          f"(шагов: {actual_steps}, skip={pct:.4f}%, "
          f"elapsed: {elapsed:.0f}s ≈ {elapsed/60:.1f}min)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--u", type=float, default=None,
                        help="одно значение u; без аргумента — все три (0.5, 1.0, 1.5) подряд")
    parser.add_argument("--out-dir", type=str, default=".",
                        help="папка для .out/.npz/.skip (по умолчанию текущая)")
    args = parser.parse_args()
    us = (args.u,) if args.u is not None else (0.5, 1.0, 1.5)
    for uv in us:
        run(uv, args.out_dir)
