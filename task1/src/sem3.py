# -*- coding: utf-8 -*-
"""Семинар 3 (task1) — консервативная проекционная схема Черемисина для
РЕЛАКСАЦИИ СМЕСИ двух газов одинаковой массы.

Задача: начальное состояние — сумма двух максвелловских распределений
холодного газа (T_c = 1) и горячего газа (T_h = 2). Прослеживаем эволюцию
к общему равновесию (T = 1.5) и считаем парциальные/общие T, n, S.

Метод (новое Пособие, uchebnoe-posobie.pdf):
  • Глава 3 — проекционный метод для простого газа, сетки Коробова,
    8-кратная симметрия по знакам компонент скорости.
  • Глава 4 — обобщение на СМЕСИ газов. Для частиц одинаковой массы
    динамика столкновения не зависит от «сорта» частицы, поэтому каждая
    компонента релаксирует по уравнению
        ∂f_c/∂t = J(f_c, f_tot),   f_tot = f_c + f_h,
    где «пробная» частица берётся из f_c, а партнёр — из полной f_tot
    (частота столкновений определяется суммарной концентрацией).

Ключевые свойства схемы (отвечают на замечание преподавателя «схема Эйлера
не работает»):
  • Логарифмическая (геометрическая) интерполяция разлётных значений
        f'_α ≈ (f_λ f_μ)^{1-r} (f_{λ+s} f_{μ+p})^r
    обеспечивает положительность f и H-теорему (это и есть «работающая
    схема из Пособия», а не простой явный Эйлер).
  • Прирост в узле α компенсируется убылью в проекционных узлах λ, λ+s
    (для партнёра — μ, μ+p), что даёт ТОЧНОЕ сохранение числа частиц
    каждой компоненты, суммарного импульса и суммарной энергии.

Главные функции:
  initialize_distribution(N, xi_cut)            → f_c, f_h, xi_grid, dxi
  compute_collision_integral(f_c, f_h, ...)     → f_c_new, f_h_new [, stats]
  compute_macro_parameters(f, xi_grid, dxi)     → n, T, S
"""
import math
from sem2 import *  # numpy as np, create_velocity_grid, ...


# === Сетки Коробова (Пособие, Глава 3) ====================================
def eratosthenes_sieve(n):
    """Простые числа до n (решето Эратосфена)."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(np.sqrt(n)) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(n + 1) if sieve[i]]


def H(z, p, s=5):
    """Функция H(z) для подбора коэффициентов сетки Коробова."""
    total = 0.0
    for k in range(1, (p - 1) // 2 + 1):
        product = 1.0
        for i in range(1, s + 1):
            kz = k
            for _ in range(i):
                kz = (kz * z) % p
            frac = kz / p
            product *= (1 - 2 * (frac - int(frac))) ** 2
        total += product
    return total


def find_korobov_coefficients(p):
    """Оптимальные коэффициенты сетки Коробова для простого p."""
    H_values = [(H(z, p), z) for z in range(1, (p - 1) // 2 + 1)]
    min_H = min(H_values, key=lambda x: x[0])[0]
    b = [h[1] for h in H_values if h[0] == min_H][0]
    a = [1]
    current = b
    for _ in range(1, p):
        a.append(current)
        current = (current * b) % p
    return a


# === Начальное состояние ==================================================
def initialize_distribution(N, xi_cut):
    """Начальное состояние смеси: f = f_c(T=1) + f_h(T=2).

    Нормировка: n_c = n_h = 1/2 (то есть n_tot = 1). Вне сферы обрезания
    ξ_cut функция распределения зануляется (чтобы в расчётной области
    f > 0 всюду — необходимо для геометрической интерполяции)."""
    xi_grid, dxi = create_velocity_grid(N, N, N, xi_cut)
    X, Y, Z = np.meshgrid(xi_grid[0], xi_grid[1], xi_grid[2], indexing="ij")
    xi_sq = X ** 2 + Y ** 2 + Z ** 2
    R = np.sqrt(xi_sq)

    T_c, T_h = 1.0, 2.0
    f_c = (2 * np.pi * T_c) ** (-1.5) * np.exp(-0.5 * xi_sq / T_c)
    f_h = (2 * np.pi * T_h) ** (-1.5) * np.exp(-0.5 * xi_sq / T_h)
    f_c[R > xi_cut] = 0.0
    f_h[R > xi_cut] = 0.0

    volume = dxi[0] * dxi[1] * dxi[2]
    f_c *= 0.5 / (np.sum(f_c) * volume)  # n_c = 0.5
    f_h *= 0.5 / (np.sum(f_h) * volume)  # n_h = 0.5
    return f_c, f_h, xi_grid, dxi


# === Проекционные узлы (сохранение импульса + энергии) ====================
def find_interpolation_nodes(xi_prime, alpha_idx, beta_idx, xi_grid, dxi, xi_cut):
    """Находит проекционные узлы λ, λ+s (для разлётной скорости ξ') и μ, μ+p
    (для партнёра ξ'_1) и коэффициент r.

    Импульс сохраняется ТОЧНО: μ = α + β − λ (в индексах сетки).
    Энергия сохраняется выбором r из условия
        (1−r)·E_near + r·E_far = E_0,   E_0 = ξ_α² + ξ_β².
    Возвращает (idx_l, idx_m, idx_ls, idx_ms, r) или None, если подходящих
    узлов внутри сферы обрезания нет."""
    N = len(xi_grid[0])

    ix0 = int(np.floor((xi_prime[0] - xi_grid[0][0]) / dxi[0]))
    iy0 = int(np.floor((xi_prime[1] - xi_grid[1][0]) / dxi[1]))
    iz0 = int(np.floor((xi_prime[2] - xi_grid[2][0]) / dxi[2]))

    valid_corners = []
    xi_cut_sq = xi_cut ** 2

    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                cx, cy, cz = ix0 + dx, iy0 + dy, iz0 + dz
                if 0 <= cx < N and 0 <= cy < N and 0 <= cz < N:
                    # Сохранение импульса: μ = α + β − λ
                    mx = alpha_idx[0] + beta_idx[0] - cx
                    my = alpha_idx[1] + beta_idx[1] - cy
                    mz = alpha_idx[2] + beta_idx[2] - cz
                    if 0 <= mx < N and 0 <= my < N and 0 <= mz < N:
                        v_cx, v_cy, v_cz = xi_grid[0][cx], xi_grid[1][cy], xi_grid[2][cz]
                        v_mx, v_my, v_mz = xi_grid[0][mx], xi_grid[1][my], xi_grid[2][mz]
                        norm_c_sq = v_cx ** 2 + v_cy ** 2 + v_cz ** 2
                        norm_m_sq = v_mx ** 2 + v_my ** 2 + v_mz ** 2
                        if norm_c_sq <= xi_cut_sq and norm_m_sq <= xi_cut_sq:
                            E_c = norm_c_sq + norm_m_sq
                            dist = ((v_cx - xi_prime[0]) ** 2 + (v_cy - xi_prime[1]) ** 2 + (v_cz - xi_prime[2]) ** 2)
                            valid_corners.append((dist, E_c, (cx, cy, cz), (mx, my, mz)))

    if not valid_corners:
        return None

    valid_corners.sort(key=lambda x: x[0])
    E_near = valid_corners[0][1]
    idx_near = valid_corners[0][2]
    idx_near_m = valid_corners[0][3]

    v_ax, v_ay, v_az = xi_grid[0][alpha_idx[0]], xi_grid[1][alpha_idx[1]], xi_grid[2][alpha_idx[2]]
    v_bx, v_by, v_bz = xi_grid[0][beta_idx[0]], xi_grid[1][beta_idx[1]], xi_grid[2][beta_idx[2]]
    E_0 = v_ax ** 2 + v_ay ** 2 + v_az ** 2 + v_bx ** 2 + v_by ** 2 + v_bz ** 2

    if abs(E_0 - E_near) < 1e-12:
        return idx_near, idx_near_m, idx_near, idx_near_m, 0.0

    if E_0 < E_near:
        candidates = [c for c in valid_corners if c[1] <= E_0]
        if not candidates:
            return None
        c_best = candidates[0]
        E_1 = c_best[1]
        if abs(E_1 - E_near) < 1e-12:
            return None
        return c_best[2], c_best[3], idx_near, idx_near_m, (E_0 - E_1) / (E_near - E_1)
    else:
        candidates = [c for c in valid_corners if c[1] >= E_0]
        if not candidates:
            return None
        c_best = candidates[0]
        E_2 = c_best[1]
        if abs(E_near - E_2) < 1e-12:
            return None
        return idx_near, idx_near_m, c_best[2], c_best[3], (E_0 - E_near) / (E_2 - E_near)


# === Интеграл столкновений для смеси ======================================
def compute_collision_integral(f_c, f_h, xi_grid, dxi, xi_cut, tau,
                               b_max=1.0, return_stats=False):
    """Один шаг по времени для смеси: возвращает обновлённые (f_c_new, f_h_new).

    Для каждой точки 8-мерной сетки Коробова разыгрывается столкновение
    пары узлов (α, β). Пробная частица каждой компоненты «переезжает»
    α → λ, партнёр (полная f_tot) — β → μ. Логарифмическая интерполяция и
    8-кратная симметрия применяются к ОБЕИМ компонентам одновременно.

    return_stats=True → дополнительно (n_pos_skipped, p)."""
    N = f_c.shape[0]
    volume = dxi[0] * dxi[1] * dxi[2]
    V_sph = (4 / 3) * np.pi * xi_cut ** 3
    N_0 = int(round(V_sph / volume))

    # 8-мерная сетка Коробова (Пособие, Глава 3, p = 50021): ε, b² и шесть
    # компонент скоростей обеих частиц берутся из одной сетки.
    p = 50021
    a = [1, 11281, 7537, 39218, 32534, 11977, 5816, 32765]

    # Коэффициент перед суммой (формула 3.26). Знаменатель — N_ν, число точек
    # сетки Коробова, у которых ОБЕ скорости попали в сферу обрезания:
    #   N_ν ≈ p·(π/6)² ≈ 13700  (для p = 50021).
    N_v = int(round(p * (np.pi / 6) ** 2))
    C = (b_max ** 2 * V_sph * N_0) / (4 * np.sqrt(2) * N_v) * tau

    shift = np.random.rand(8)

    f_c_new = f_c.copy()
    f_h_new = f_h.copy()
    n_pos_skipped = 0

    for i in range(1, p + 1):
        u = [(((i * a[j]) % p) / p + shift[j]) for j in range(8)]
        u = [v - int(v) for v in u]
        u1, u2, u3, u4, u5, u6, u7, u8 = u

        epsilon = u1 * 2 * np.pi
        S_b = u2 * (b_max ** 2)

        # Первая скорость (узел α)
        v0x = u3 * 2 * xi_cut - xi_cut
        v0y = u4 * 2 * xi_cut - xi_cut
        v0z = u5 * 2 * xi_cut - xi_cut
        if v0x ** 2 + v0y ** 2 + v0z ** 2 > xi_cut ** 2:
            continue
        alpha_idx = (
            max(0, min(N - 1, int(np.floor((v0x - xi_grid[0][0]) / dxi[0] + 0.5)))),
            max(0, min(N - 1, int(np.floor((v0y - xi_grid[1][0]) / dxi[1] + 0.5)))),
            max(0, min(N - 1, int(np.floor((v0z - xi_grid[2][0]) / dxi[2] + 0.5)))),
        )
        v_ax, v_ay, v_az = xi_grid[0][alpha_idx[0]], xi_grid[1][alpha_idx[1]], xi_grid[2][alpha_idx[2]]
        if v_ax ** 2 + v_ay ** 2 + v_az ** 2 > xi_cut ** 2:
            continue

        # Вторая скорость (узел β)
        v1x = u6 * 2 * xi_cut - xi_cut
        v1y = u7 * 2 * xi_cut - xi_cut
        v1z = u8 * 2 * xi_cut - xi_cut
        if v1x ** 2 + v1y ** 2 + v1z ** 2 > xi_cut ** 2:
            continue
        beta_idx = (
            max(0, min(N - 1, int(np.floor((v1x - xi_grid[0][0]) / dxi[0] + 0.5)))),
            max(0, min(N - 1, int(np.floor((v1y - xi_grid[1][0]) / dxi[1] + 0.5)))),
            max(0, min(N - 1, int(np.floor((v1z - xi_grid[2][0]) / dxi[2] + 0.5)))),
        )
        v_bx, v_by, v_bz = xi_grid[0][beta_idx[0]], xi_grid[1][beta_idx[1]], xi_grid[2][beta_idx[2]]
        if v_bx ** 2 + v_by ** 2 + v_bz ** 2 > xi_cut ** 2:
            continue

        gx, gy, gz = v_bx - v_ax, v_by - v_ay, v_bz - v_az
        g_norm_sq = gx ** 2 + gy ** 2 + gz ** 2
        if g_norm_sq < 1e-10:
            continue
        g_norm = np.sqrt(g_norm_sq)

        b = np.sqrt(S_b)
        if b > 1.0:
            continue
        theta = 2 * math.acos(b)

        cos_t, sin_t = np.cos(theta), np.sin(theta)
        cos_e, sin_e = np.cos(epsilon), np.sin(epsilon)
        g_xy = np.sqrt(gx ** 2 + gy ** 2)
        if g_xy > 1e-10:
            gx_new = gx * cos_t - (gx * gz / g_xy) * cos_e * sin_t + (g_norm * gy / g_xy) * sin_e * sin_t
            gy_new = gy * cos_t - (gy * gz / g_xy) * cos_e * sin_t - (g_norm * gx / g_xy) * sin_e * sin_t
            gz_new = gz * cos_t + g_xy * cos_e * sin_t
        else:
            gx_new = g_norm * sin_t * sin_e
            gy_new = g_norm * sin_t * cos_e
            gz_new = g_norm * cos_t

        xi_prime = (
            0.5 * (v_ax + v_bx) - 0.5 * gx_new,
            0.5 * (v_ay + v_by) - 0.5 * gy_new,
            0.5 * (v_az + v_bz) - 0.5 * gz_new,
        )
        if xi_prime[0] ** 2 + xi_prime[1] ** 2 + xi_prime[2] ** 2 > xi_cut ** 2:
            continue

        nodes = find_interpolation_nodes(xi_prime, alpha_idx, beta_idx, xi_grid, dxi, xi_cut)
        if nodes is None:
            continue
        idx_l, idx_m, idx_ls, idx_ms, r_v = nodes

        # Значения ф.р. в узлах столкновения (отрицательные обрезаем нулём —
        # нужно для возведения в дробную степень при геом. интерполяции).
        fc_a, fc_b = f_c_new[alpha_idx], f_c_new[beta_idx]
        fh_a, fh_b = f_h_new[alpha_idx], f_h_new[beta_idx]
        ftot_a, ftot_b = fc_a + fh_a, fc_b + fh_b

        fc_l = max(0.0, f_c_new[idx_l]); fc_m = max(0.0, f_c_new[idx_m])
        fc_ls = max(0.0, f_c_new[idx_ls]); fc_ms = max(0.0, f_c_new[idx_ms])
        fh_l = max(0.0, f_h_new[idx_l]); fh_m = max(0.0, f_h_new[idx_m])
        fh_ls = max(0.0, f_h_new[idx_ls]); fh_ms = max(0.0, f_h_new[idx_ms])
        ftot_l = fc_l + fh_l; ftot_m = fc_m + fh_m
        ftot_ls = fc_ls + fh_ls; ftot_ms = fc_ms + fh_ms

        # Геометрическая интерполяция парных произведений в точках разлёта.
        # α-слот: пробная частица α↔λ, партнёр (f_tot) β↔μ.
        gain_c_a = (fc_l * ftot_m) ** (1 - r_v) * (fc_ls * ftot_ms) ** r_v
        gain_h_a = (fh_l * ftot_m) ** (1 - r_v) * (fh_ls * ftot_ms) ** r_v
        # β-слот: пробная частица β↔μ, партнёр (f_tot) α↔λ.
        gain_c_b = (ftot_l * fc_m) ** (1 - r_v) * (ftot_ls * fc_ms) ** r_v
        gain_h_b = (ftot_l * fh_m) ** (1 - r_v) * (ftot_ls * fh_ms) ** r_v

        Om_c_a = (gain_c_a - fc_a * ftot_b) * g_norm
        Om_h_a = (gain_h_a - fh_a * ftot_b) * g_norm
        Om_c_b = (gain_c_b - ftot_a * fc_b) * g_norm
        Om_h_b = (gain_h_b - ftot_a * fh_b) * g_norm

        dca, dha = C * Om_c_a, C * Om_h_a
        dcb, dhb = C * Om_c_b, C * Om_h_b

        # Контроль положительности (по канонической паре узлов; полная дельта —
        # консервативная оценка, реально на узел приходится дельта/8).
        if (fc_a + dca < 0 or fh_a + dha < 0 or
                fc_b + dcb < 0 or fh_b + dhb < 0 or
                f_c_new[idx_l] - (1 - r_v) * dca < 0 or f_h_new[idx_l] - (1 - r_v) * dha < 0 or
                f_c_new[idx_ls] - r_v * dca < 0 or f_h_new[idx_ls] - r_v * dha < 0 or
                f_c_new[idx_m] - (1 - r_v) * dcb < 0 or f_h_new[idx_m] - (1 - r_v) * dhb < 0 or
                f_c_new[idx_ms] - r_v * dcb < 0 or f_h_new[idx_ms] - r_v * dhb < 0):
            n_pos_skipped += 1
            continue

        # === 8-кратная симметрия по знакам (ξ_x, ξ_y, ξ_z) ===
        # Каждое столкновение представляет 8 эквивалентных событий, поэтому
        # дельта делится на 8 и раскладывается по всем зеркальным копиям.
        node_updates = (
            (alpha_idx, dca, dha),
            (beta_idx, dcb, dhb),
            (idx_l, -(1 - r_v) * dca, -(1 - r_v) * dha),
            (idx_ls, -r_v * dca, -r_v * dha),
            (idx_m, -(1 - r_v) * dcb, -(1 - r_v) * dhb),
            (idx_ms, -r_v * dcb, -r_v * dhb),
        )
        for (ix, iy, iz), dc, dh in node_updates:
            dc8, dh8 = dc / 8.0, dh / 8.0
            for sx in (1, -1):
                jx = ix if sx == 1 else N - 1 - ix
                for sy in (1, -1):
                    jy = iy if sy == 1 else N - 1 - iy
                    for sz in (1, -1):
                        jz = iz if sz == 1 else N - 1 - iz
                        f_c_new[jx, jy, jz] += dc8
                        f_h_new[jx, jy, jz] += dh8

    if return_stats:
        return f_c_new, f_h_new, n_pos_skipped, p
    return f_c_new, f_h_new


# === Макропараметры =======================================================
def compute_macro_parameters(f, xi_grid, dxi):
    """Возвращает (n, T, S) для одной компоненты или для смеси.

    n — концентрация ∫f dξ;
    T — температура (1/3)∫f|ξ−u|² dξ / n  (для покоящегося газа u = 0);
    S — энтропия −∫ f ln f dξ  (кинетическое определение, H = ∫f ln f)."""
    volume = dxi[0] * dxi[1] * dxi[2]
    X, Y, Z = np.meshgrid(xi_grid[0], xi_grid[1], xi_grid[2], indexing="ij")

    n = np.sum(f) * volume
    if n <= 0:
        return 0.0, 0.0, 0.0

    u_x = np.sum(f * X) * volume / n
    u_y = np.sum(f * Y) * volume / n
    u_z = np.sum(f * Z) * volume / n

    T = np.sum(f * ((X - u_x) ** 2 + (Y - u_y) ** 2 + (Z - u_z) ** 2)) * volume / (3 * n)

    mask = f > 1e-300
    S = -np.sum(f[mask] * np.log(f[mask])) * volume
    return n, T, S
