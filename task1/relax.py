# -*- coding: utf-8 -*-
from sem3 import *

# --- Настройки сетки и задачи ---
N = 20
xi_cut = 4.8
tau = 0.01
time_steps = 1000 # Количество шагов до равновесия

# Инициализация сетки и распределений (холодный и горячий газ)
f_c, f_h, xi_grid, dxi = initialize_distribution(N, xi_cut)

# Расчет количества узлов на сетке
N_0 = 4224
W_min = (N_0 * xi_cut ** 4) / (6 * np.sqrt(np.pi))
N_v = int(W_min * tau)

# Основной цикл симуляции
for t in range(time_steps):
    f_tot = f_c + f_h

    # Расчет макропараметров
    n_c, T_c, S_c = compute_macro_parameters(f_c, xi_grid, dxi)
    n_h, T_h, S_h = compute_macro_parameters(f_h, xi_grid, dxi)
    n_tot, T_tot, S_tot = compute_macro_parameters(f_tot, xi_grid, dxi)

    # Требуемый формат вывода
    if t % 10 == 0:
        print(f"{T_tot:.5f} {T_c:.5f} {T_h:.5f} {n_c:.5f} {n_h:.5f} {S_tot:.5f} {S_c:.5f} {S_h:.5f}")

    if np.allclose([T_c, T_h, T_tot], [1.5, 1.5, 1.5], rtol=1e-3): break
    # Вычисление интеграла столкновений (эволюция компонент во времени)
    f_c, f_h = compute_collision_integral(f_c, f_h, xi_grid, dxi, xi_cut, N_v, tau=tau)