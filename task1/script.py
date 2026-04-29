# -*- coding: utf-8 -*-
from sem3 import *

xi_p, xi1_p = simulate_collisions(check_conservation=True, save_to_file=False)

N = 20
xi_cut = 4.8

(xi_x, xi_y, xi_z), dxi = create_velocity_grid(N, N, N, xi_cut)

a = create_random_params(20000)
normalize_params(a, xi_cut)

a_snapped, snapped_idx = snap_collision_velocities(a, (xi_x, xi_y, xi_z), dxi)

a_snapped_filtered, snapped_idx_filtered = filter_collision_nodes(a_snapped, snapped_idx, xi_cut)

a_updated = compute_post_collision_velocities_sem_2(a_snapped_filtered, 1.0)

lam1, lam2, mu1, mu2, r = find_interpolating_nodes_and_weights(a_updated, (xi_x, xi_y, xi_z), dxi, xi_cut)

tau = 0.02
T_tilde = 0.95
time_steps = int(0.4 / tau)
# f_max = 1 / ((2 * pi) ** 1.5)
# V_sph = 4 * pi * xi_cut ** 3 / 3
N_0 = 4224
W_min = (N_0 * xi_cut ** 4) / (6 * sqrt(pi))
N_v = int(W_min * tau)
f, xi_grid, dxi = initialize_distribution(N, xi_cut, T_tilde, condition=1)
print("initialize_distribution done")
n, u, T = compute_macro_parameters(f, xi_grid, dxi)
print("compute_macro_parameters done")
u_star, T_star = newton_method(xi_grid, dxi, u, T)
print("newton_method done")
f_M = np.exp(-0.5 * (xi_grid - u_star) ** 2 / T_star)
volume = dxi[0] * dxi[1] * dxi[2]
f_M /= np.sum(f_M, axis=0) * volume
# print(np.sum((f - f_M) ** 2))
initial_norms = compute_norms(f, f_M, xi_grid, dxi, xi_cut)
print("compute_norms done")
delta_k = []
print(f"delta_k computing starting")
for t in range(time_steps):
    # Добавляем номер шага в вывод
    print(f"\n--- Step {t + 1}/{time_steps} ---")
    # print(t + 1)

    # Вычисление интеграла столкновений
    f = compute_collision_integral(f, xi_grid, dxi, xi_cut, N_v, tau=tau)
    print(f"compute_collision_integral done")

    # Вычисление норм
    norms = compute_norms(f, f_M, xi_grid, dxi, xi_cut)
    print(f"compute_norms done")

    # Сохранение результатов
    delta_k.append([norm / init_norm for norm, init_norm in zip(norms, initial_norms)])

print("delta_k computed", delta_k)
# Conservation check
n_new, u_new, T_new = compute_macro_parameters(f, xi_grid, dxi)
print(f"Conservation check: Δn={abs(n_new - n)}, Δu={np.linalg.norm(u_new - u)}, ΔT={abs(T_new - T)}")

# Symmetry check
f_0 = f.copy()
tau_sym = 1e-6
time_steps_sym = int(100 * tau_sym / tau)
arr_f = []
for _ in range(time_steps_sym):
    f = compute_collision_integral(f, xi_grid, dxi, xi_cut, N_v, tau=tau_sym)
    arr_f.append(f)

print(arr_f)
I = (f - f_0) / (100 * tau_sym)
sym_error = sqrt(np.sum((I[:N // 2, :, :] - I[N - 1:N // 2 - 1:-1, :, :]) ** 2)) / sqrt(0.5 * np.sum(I ** 2))
print(f"Symmetry error: {sym_error}")

# Convergence and formal checks would require additional runs with different N and p
# Plotting delta_k vs model solution

# t = np.array([i * tau for i in range(time_steps)])
# model = -t * 16 / (5 * sqrt(2 * pi / T))

# Repeat for t=2 and condition=2 as needed
