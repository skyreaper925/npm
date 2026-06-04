# -*- coding: utf-8 -*-
"""Построение графиков для отчёта task1 (релаксация смеси газов).

Канонический источник кода для графиков. Из него же генерируется
relax.ipynb (см. build_notebook.py). Запускать из папки data/:

    python make_report_plots.py

Читает mixture.out (8 столбцов) и f_mixture.npz, кладёт PNG в data/img/.
Файлы с разметкой `# %%` — границы ячеек будущего блокнота.
"""
# %% [markdown]
# # Релаксация смеси «холодный (T=1) + горячий (T=2)» газ
#
# Строим графики по результатам `relax.py`:
# `mixture.out` (T, n, S по шагам) и `f_mixture.npz` (снимки f_c, f_h).

# %% Загрузка данных
import io
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # без дисплея (рендер в файлы)
import matplotlib.pyplot as plt
from pathlib import Path

# Windows-консоль cp1251 не печатает кириллицу/спецсимволы — делаем stdout UTF-8.
if hasattr(sys.stdout, "buffer") and (getattr(sys.stdout, "encoding", "") or "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Находим папку с данными (data/) независимо от текущей директории.
_CANDS = [Path("."), Path("data"), Path("../data"),
          Path(__file__).resolve().parent if "__file__" in globals() else Path(".")]
DATA_DIR = next((c for c in _CANDS if (c / "mixture.out").exists()), Path("."))
IMG_DIR = DATA_DIR / "img"
IMG_DIR.mkdir(exist_ok=True)
print("DATA_DIR =", DATA_DIR.resolve())

arr = np.loadtxt(DATA_DIR / "mixture.out")
if arr.ndim == 1:
    arr = arr.reshape(1, -1)
# Столбцы: T_tot T_c T_h n_c n_h S_tot S_c S_h
T_tot, T_c, T_h, n_c, n_h, S_tot, S_c, S_h = arr.T
steps = np.arange(len(arr))
T_eq = T_tot[-1]  # общая равновесная температура (сохраняется ~ const)
print(f"шагов: {len(arr)};  T_eq≈{T_eq:.4f};  "
      f"T_c: {T_c[0]:.3f}->{T_c[-1]:.3f};  T_h: {T_h[0]:.3f}->{T_h[-1]:.3f}")

npz = np.load(DATA_DIR / "f_mixture.npz")
snap_c = npz["snapshots_c"]       # (n_snap, N, N, N)
snap_h = npz["snapshots_h"]
t_snap = npz["t_snapshots"]
xi_x = npz["xi_x"]
dxi = float(xi_x[1] - xi_x[0])

# %% Рис. 1 — релаксация температур
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(steps, T_c, color="tab:blue", label=r"$T_c$ (холодный)")
ax.plot(steps, T_h, color="tab:red", label=r"$T_h$ (горячий)")
ax.plot(steps, T_tot, color="tab:green", lw=2.2, label=r"$T$ (смесь)")
ax.axhline(T_eq, ls="--", color="gray", lw=1, label=fr"равновесие $T={T_eq:.3f}$")
ax.set_xlabel("шаг по времени $n$ (t = nτ)")
ax.set_ylabel("температура")
ax.set_title("Релаксация температур: $T_c\\uparrow$, $T_h\\downarrow$ к общему значению")
ax.legend()
ax.grid(alpha=0.3)
fig.savefig(IMG_DIR / "T_combined.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("[ok] T_combined.png")

# %% Рис. 2 — сохранение концентраций (контроль)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(steps, n_c, color="tab:blue", label=r"$n_c$ (холодный)")
ax.plot(steps, n_h, color="tab:red", label=r"$n_h$ (горячий)")
ax.plot(steps, n_c + n_h, color="tab:green", lw=2, label=r"$n = n_c + n_h$")
ax.set_ylim(0.0, 1.1)
ax.set_xlabel("шаг по времени $n$")
ax.set_ylabel("концентрация")
ax.set_title("Сохранение концентраций компонент (контроль): $n_c=n_h=0.5$")
ax.legend()
ax.grid(alpha=0.3)
fig.savefig(IMG_DIR / "n_combined.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("[ok] n_combined.png")

# %% Рис. 3 — энтропия (H-теорема)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(steps, S_c, color="tab:blue", label=r"$S_c$ (холодный)")
ax.plot(steps, S_h, color="tab:red", label=r"$S_h$ (горячий)")
ax.plot(steps, S_tot, color="tab:green", lw=2.2, label=r"$S$ (смесь)")
ax.set_xlabel("шаг по времени $n$")
ax.set_ylabel(r"энтропия $S = -\int f\ln f\, d\xi$")
ax.set_title("Энтропия: $S_c\\uparrow$, $S_h\\downarrow$, а $S$ смеси не убывает (H-теорема)")
ax.legend(loc="center right")
ax.grid(alpha=0.3)
# Вставка: увеличенный S смеси — видно монотонный рост (H-теорема).
axins = ax.inset_axes([0.30, 0.45, 0.42, 0.30])
axins.plot(steps, S_tot, color="tab:green", lw=2)
axins.set_title("$S$ смеси крупно: $S\\uparrow$", fontsize=8)
axins.ticklabel_format(axis="y", style="plain", useOffset=False)
axins.tick_params(labelsize=7)
axins.grid(alpha=0.3)
fig.savefig(IMG_DIR / "S_combined.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("[ok] S_combined.png")

# %% Рис. 4 — экспоненциальное приближение к равновесию
fig, ax = plt.subplots(figsize=(8, 5))
gap = np.abs(T_c - T_h)
ax.semilogy(steps, np.maximum(gap, 1e-12), color="tab:purple", lw=2,
            label=r"$|T_c - T_h|$")
ax.semilogy(steps, np.maximum(np.abs(T_c - T_eq), 1e-12), color="tab:blue", ls="--",
            label=r"$|T_c - T_{eq}|$")
ax.semilogy(steps, np.maximum(np.abs(T_h - T_eq), 1e-12), color="tab:red", ls=":",
            label=r"$|T_h - T_{eq}|$")
ax.set_xlabel("шаг по времени $n$")
ax.set_ylabel("отклонение от равновесия (лог. шкала)")
ax.set_title("Экспоненциальная релаксация разности температур")
ax.legend()
ax.grid(alpha=0.3, which="both")
fig.savefig(IMG_DIR / "anisotropy_log.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("[ok] anisotropy_log.png")


# %% Рис. 5 — эволюция маргинальных распределений f_c, f_h
def marginal_x(f3d):
    """f(ξ_x) = ∫∫ f dξ_y dξ_z."""
    return f3d.sum(axis=(1, 2)) * dxi ** 2


for snaps, label, fname, color0 in [
    (snap_c, "f_c", "f_marginal_c.png", "Blues"),
    (snap_h, "f_h", "f_marginal_h.png", "Reds"),
]:
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap(color0)
    n_s = len(t_snap)
    for k in range(n_s):
        frac = 0.3 + 0.7 * k / max(1, n_s - 1)
        ax.plot(xi_x, marginal_x(snaps[k]), color=cmap(frac),
                label=f"$n={int(t_snap[k])}$")
    ax.set_xlabel(r"$\xi_x$")
    ax.set_ylabel(fr"маргинальная ${label}(\xi_x)$")
    ax.set_title(fr"Эволюция ${label}$: от начальной к равновесной максвелловской")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.savefig(IMG_DIR / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {fname}")

# %% Рис. 6 — двумерные срезы f_tot в плоскости (ξ_x, ξ_y) в начале и конце
def proj_xy(f3d):
    """f(ξ_x, ξ_y) = ∫ f dξ_z."""
    return f3d.sum(axis=2) * dxi

for snaps_pair, tag, fname in [
    ((snap_c[0], snap_h[0]), f"n={int(t_snap[0])} (начало)", "f_xy_initial.png"),
    ((snap_c[-1], snap_h[-1]), f"n={int(t_snap[-1])} (конец)", "f_xy_final.png"),
]:
    fc_xy = proj_xy(snaps_pair[0])
    fh_xy = proj_xy(snaps_pair[1])
    ftot_xy = fc_xy + fh_xy
    fig, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contourf(xi_x, xi_x, ftot_xy.T, levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax, label=r"$\int f_{tot}\, d\xi_z$")
    ax.set_xlabel(r"$\xi_x$")
    ax.set_ylabel(r"$\xi_y$")
    ax.set_aspect("equal")
    ax.set_title(f"Полная ф.р. смеси, {tag}")
    fig.savefig(IMG_DIR / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {fname}")

# %%
print("Все графики сохранены в", IMG_DIR.resolve())
