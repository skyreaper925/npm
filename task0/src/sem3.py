# -*- coding: utf-8 -*-
import math
from sem2 import *

def eratosthenes_sieve(n):
    """Generate prime numbers up to n using Sieve of Eratosthenes."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(np.sqrt(n)) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(n + 1) if sieve[i]]


def H(z, p, s=5):
    """Compute H(z) function for Korobov grid coefficients (Eq. 3.35)."""
    total = .0
    for k in range(1, (p - 1) // 2 + 1):
        product = 1.0
        for i in range(1, s + 1):
            # Iterative modulo to avoid overflow
            kz = k
            for _ in range(i):
                kz = (kz * z) % p
            frac = kz / p
            product *= (1 - 2 * (frac - int(frac))) ** 2

        total += product

    return total


def find_korobov_coefficients(p):
    """Find Korobov grid coefficients for prime p."""
    H_values = [(H(z, p), z) for z in range(1, (p - 1) // 2 + 1)]
    min_H = min(H_values, key=lambda x: x[0])[0]
    b = [h[1] for h in H_values if h[0] == min_H][0]
    a = [1]
    current = b
    for _ in range(1, p):
        a.append(current)
        current = (current * b) % p
    return a


def initialize_distribution(N, xi_cut, u):
    xi_grid, dxi = create_velocity_grid(N, N, N, xi_cut)
    X, Y, Z = np.meshgrid(xi_grid[0], xi_grid[1], xi_grid[2], indexing='ij')

    # Bimodal
    f = 0.5 * (2 * np.pi)**(-1.5) * (
        np.exp(-0.5 * ((X - u)**2 + Y**2 + Z**2)) +
        np.exp(-0.5 * ((X + u)**2 + Y**2 + Z**2))
    )

    # Обрезаем все, что за пределами сферы xi_cut
    R = np.sqrt(X**2 + Y**2 + Z**2)
    f[R > xi_cut] = 0.0

    # Численная нормировка (n = 1.0)
    volume = dxi[0] * dxi[1] * dxi[2]
    n = np.sum(f) * volume
    f /= n

    return f, xi_grid, dxi

def find_interpolation_nodes(xi_prime, alpha_idx, beta_idx, xi_grid, dxi, xi_cut):
    N = len(xi_grid[0])

    # Координаты ячейки для xi_prime
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
                    # Строгое сохранение импульса: mu = alpha + beta - lambda
                    mx = alpha_idx[0] + beta_idx[0] - cx
                    my = alpha_idx[1] + beta_idx[1] - cy
                    mz = alpha_idx[2] + beta_idx[2] - cz

                    if 0 <= mx < N and 0 <= my < N and 0 <= mz < N:
                        v_cx, v_cy, v_cz = xi_grid[0][cx], xi_grid[1][cy], xi_grid[2][cz]
                        v_mx, v_my, v_mz = xi_grid[0][mx], xi_grid[1][my], xi_grid[2][mz]

                        norm_c_sq = v_cx**2 + v_cy**2 + v_cz**2
                        norm_m_sq = v_mx**2 + v_my**2 + v_mz**2

                        if norm_c_sq <= xi_cut_sq and norm_m_sq <= xi_cut_sq:
                            E_c = norm_c_sq + norm_m_sq
                            dist = (v_cx - xi_prime[0])**2 + (v_cy - xi_prime[1])**2 + (v_cz - xi_prime[2])**2
                            valid_corners.append((dist, E_c, (cx, cy, cz), (mx, my, mz)))

    if not valid_corners:
        return None

    # eta_near - ближайший узел к скорости
    valid_corners.sort(key=lambda x: x[0])
    E_near = valid_corners[0][1]
    idx_near = valid_corners[0][2]
    idx_near_m = valid_corners[0][3]

    v_ax, v_ay, v_az = xi_grid[0][alpha_idx[0]], xi_grid[1][alpha_idx[1]], xi_grid[2][alpha_idx[2]]
    v_bx, v_by, v_bz = xi_grid[0][beta_idx[0]], xi_grid[1][beta_idx[1]], xi_grid[2][beta_idx[2]]
    E_0 = v_ax**2 + v_ay**2 + v_az**2 + v_bx**2 + v_by**2 + v_bz**2

    if abs(E_0 - E_near) < 1e-12:
        return idx_near, idx_near_m, idx_near, idx_near_m, 0.0

    if E_0 < E_near:
        # idx_ls, idx_ms = idx_near, idx_near_m
        # E_2 = E_near
        candidates = [c for c in valid_corners if c[1] <= E_0]
        if not candidates: return None
        c_best = candidates[0]
        # idx_l, idx_m = c_best[2], c_best[3]
        E_1 = c_best[1]
        if abs(E_1 - E_near) < 1e-12: return None
        return c_best[2], c_best[3], idx_near, idx_near_m, (E_0 - E_1) / (E_near - E_1)
    else:
        # idx_l, idx_m = idx_near, idx_near_m
        # E_1 = E_near
        candidates = [c for c in valid_corners if c[1] >= E_0]
        if not candidates: return None
        c_best = candidates[0]
        # idx_ls, idx_ms = c_best[2], c_best[3]
        E_2 = c_best[1]
        if abs(E_near - E_2) < 1e-12: return None
        return idx_near, idx_near_m, c_best[2], c_best[3], (E_0 - E_near) / (E_2 - E_near)


def compute_collision_integral(f, xi_grid, dxi, xi_cut, tau, b_max=1.0, return_stats=False):
    """Возвращает обновлённую f. Если return_stats=True — также число пропусков
    по условию положительности (n_pos_skipped) и общее число точек Коробова (p)."""
    N = f.shape[0]
    volume = dxi[0] * dxi[1] * dxi[2]
    V_sph = (4 / 3) * np.pi * xi_cut**3
    N_0 = int(round(V_sph / volume))

    # 8-мерная сетка Коробова (Пособие, стр. 26, Таблица 1, p = 50021).
    # Шесть скоростных компонент (обе частицы) + b² + ε — все 8 параметров столкновения берутся из одной сетки
    p = 50021
    a = [1, 11281, 7537, 39218, 32534, 11977, 5816, 32765]

    # Константа перед суммой по схеме (3.22), формула (3.26) Пособия (стр. 22).
    # Знаменатель — N_ν, число точек 8-мерной сетки Коробова, у которых обе
    # скорости попадают в сферу обрезания. Аналитическая оценка:
    #   N_ν ≈ p · (V_sph / V_cube)² = p · (π/6)² ≈ 13700  (для p=50021).
    # Это **не p**: ранее в коде ошибочно стояло p в знаменателе, что давало
    # коэффициент в (π/6)² ≈ 0.274 раза меньше нужного и замедляло релаксацию
    # в ~3.6 раза.
    N_v = int(round(p * (np.pi / 6) ** 2))
    C = (b_max**2 * V_sph * N_0) / (4 * np.sqrt(2) * N_v) * tau

    shift = np.random.rand(8)

    f_new = f.copy()
    n_pos_skipped = 0  # счётчик пропусков по условию положительности (для контроля шага tau)

    for i in range(1, p + 1):
        # Генерируем 8-мерную точку сетки Коробова
        u = [(((i * a[j]) % p) / p + shift[j]) for j in range(8)]
        u = [v - int(v) for v in u]
        u1, u2, u3, u4, u5, u6, u7, u8 = u

        epsilon = u1 * 2 * np.pi
        S_b = u2 * (b_max ** 2)
        # Первая скорость (узел α)
        v0x = u3 * 2 * xi_cut - xi_cut
        v0y = u4 * 2 * xi_cut - xi_cut
        v0z = u5 * 2 * xi_cut - xi_cut
        if v0x**2 + v0y**2 + v0z**2 > xi_cut**2: continue
        alpha_idx = (
            max(0, min(N - 1, int(np.floor((v0x - xi_grid[0][0]) / dxi[0] + 0.5)))),
            max(0, min(N - 1, int(np.floor((v0y - xi_grid[1][0]) / dxi[1] + 0.5)))),
            max(0, min(N - 1, int(np.floor((v0z - xi_grid[2][0]) / dxi[2] + 0.5))))
        )
        v_ax, v_ay, v_az = xi_grid[0][alpha_idx[0]], xi_grid[1][alpha_idx[1]], xi_grid[2][alpha_idx[2]]
        if v_ax**2 + v_ay**2 + v_az**2 > xi_cut**2: continue

        # Вторая скорость (узел β)
        v1x = u6 * 2 * xi_cut - xi_cut
        v1y = u7 * 2 * xi_cut - xi_cut
        v1z = u8 * 2 * xi_cut - xi_cut
        if v1x**2 + v1y**2 + v1z**2 > xi_cut**2: continue
        beta_idx = (
            max(0, min(N - 1, int(np.floor((v1x - xi_grid[0][0]) / dxi[0] + 0.5)))),
            max(0, min(N - 1, int(np.floor((v1y - xi_grid[1][0]) / dxi[1] + 0.5)))),
            max(0, min(N - 1, int(np.floor((v1z - xi_grid[2][0]) / dxi[2] + 0.5))))
        )
        v_bx, v_by, v_bz = xi_grid[0][beta_idx[0]], xi_grid[1][beta_idx[1]], xi_grid[2][beta_idx[2]]
        if v_bx**2 + v_by**2 + v_bz**2 > xi_cut**2: continue

        gx, gy, gz = v_bx - v_ax, v_by - v_ay, v_bz - v_az
        g_norm_sq = gx**2 + gy**2 + gz**2
        if g_norm_sq < 1e-10: continue
        g_norm = np.sqrt(g_norm_sq)

        b = np.sqrt(S_b)
        if b > 1.0: continue
        theta = 2 * math.acos(b)

        # Оптимизированный поворот вектора (без создания numpy массивов)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        cos_e, sin_e = np.cos(epsilon), np.sin(epsilon)
        g_xy = np.sqrt(gx**2 + gy**2)

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
            0.5 * (v_az + v_bz) - 0.5 * gz_new
        )

        if xi_prime[0]**2 + xi_prime[1]**2 + xi_prime[2]**2 > xi_cut**2: continue

        nodes = find_interpolation_nodes(xi_prime, alpha_idx, beta_idx, xi_grid, dxi, xi_cut)
        if nodes is None: continue

        idx_l, idx_m, idx_ls, idx_ms, r_v = nodes

        f_a, f_b = f_new[alpha_idx], f_new[beta_idx]
        f_l, f_m = max(0.0, f_new[idx_l]), max(0.0, f_new[idx_m])
        f_ls, f_ms = max(0.0, f_new[idx_ls]), max(0.0, f_new[idx_ms])

        Omega = (((f_l * f_m)**(1 - r_v)) * ((f_ls * f_ms)**r_v) - f_a * f_b) * g_norm
        delta = C * Omega

        if (f_new[idx_l] - (1 - r_v) * delta < 0 or f_new[idx_ls] - r_v * delta < 0 or
            f_new[idx_m] - (1 - r_v) * delta < 0 or f_new[idx_ms] - r_v * delta < 0 or
            f_new[alpha_idx] + delta < 0 or f_new[beta_idx] + delta < 0):
            n_pos_skipped += 1
            continue

        # === Учёт симметрии ф.р. по всем 3 осям (Пособие, стр. 36-37) ===
        # Пространственно-однородная задача: f симметрична по ξ_x, ξ_y, ξ_z.
        # В методе Коробова разыгрываются скорости из всего куба, и каждое
        # столкновение «рассказывает» сразу про 8 эквивалентных событий —
        # с любой комбинацией знаков компонент скорости. Поэтому, чтобы
        # не учесть один и тот же физический вклад в 8 раз, делим delta на 8.
        # И каждый из шести узлов столкновения обновляется в каждой из 8
        # симметричных копий.
        d8 = delta / 8.0
        for sx in (1, -1):
            ix_a = alpha_idx[0] if sx == 1 else (N - 1 - alpha_idx[0])
            ix_b = beta_idx[0]  if sx == 1 else (N - 1 - beta_idx[0])
            ix_l = idx_l[0]     if sx == 1 else (N - 1 - idx_l[0])
            ix_ls= idx_ls[0]    if sx == 1 else (N - 1 - idx_ls[0])
            ix_m = idx_m[0]     if sx == 1 else (N - 1 - idx_m[0])
            ix_ms= idx_ms[0]    if sx == 1 else (N - 1 - idx_ms[0])
            for sy in (1, -1):
                iy_a = alpha_idx[1] if sy == 1 else (N - 1 - alpha_idx[1])
                iy_b = beta_idx[1]  if sy == 1 else (N - 1 - beta_idx[1])
                iy_l = idx_l[1]     if sy == 1 else (N - 1 - idx_l[1])
                iy_ls= idx_ls[1]    if sy == 1 else (N - 1 - idx_ls[1])
                iy_m = idx_m[1]     if sy == 1 else (N - 1 - idx_m[1])
                iy_ms= idx_ms[1]    if sy == 1 else (N - 1 - idx_ms[1])
                for sz in (1, -1):
                    iz_a = alpha_idx[2] if sz == 1 else (N - 1 - alpha_idx[2])
                    iz_b = beta_idx[2]  if sz == 1 else (N - 1 - beta_idx[2])
                    iz_l = idx_l[2]     if sz == 1 else (N - 1 - idx_l[2])
                    iz_ls= idx_ls[2]    if sz == 1 else (N - 1 - idx_ls[2])
                    iz_m = idx_m[2]     if sz == 1 else (N - 1 - idx_m[2])
                    iz_ms= idx_ms[2]    if sz == 1 else (N - 1 - idx_ms[2])
                    f_new[ix_a, iy_a, iz_a] += d8
                    f_new[ix_b, iy_b, iz_b] += d8
                    f_new[ix_l, iy_l, iz_l] -= (1 - r_v) * d8
                    f_new[ix_ls,iy_ls,iz_ls]-= r_v * d8
                    f_new[ix_m, iy_m, iz_m] -= (1 - r_v) * d8
                    f_new[ix_ms,iy_ms,iz_ms]-= r_v * d8

    if return_stats:
        return f_new, n_pos_skipped, p
    return f_new


def symmetrize_yz(f):
    """Симметризация f по поперечным компонентам ξ_y, ξ_z.
    НУ и оператор столкновений сохраняют симметрию по этим осям;
    стохастика метода Коробова её нарушает на уровне ~1/sqrt(p)."""
    f = 0.5 * (f + f[:, ::-1, :])
    f = 0.5 * (f + f[:, :, ::-1])
    return f


def pack_quarter(f):
    """Сжимает полный массив (N, N, N) до четверти (N, N//2, N//2),
    усредняя по симметрии ξ_y → −ξ_y и ξ_z → −ξ_z (предложение Черемисина —
    хранить ¼ ф.р.). Сетка задаётся узлами в центрах ячеек, поэтому
    f[:, ::-1, :] — точное отражение по ξ_y."""
    N = f.shape[1]
    h = N // 2
    pos_pos = f[:, h:, h:]                    # ξ_y > 0, ξ_z > 0
    neg_pos = f[:, :h, h:][:, ::-1, :]        # ξ_y < 0 → отразить
    pos_neg = f[:, h:, :h][:, :, ::-1]        # ξ_z < 0 → отразить
    neg_neg = f[:, :h, :h][:, ::-1, ::-1]     # обе → отразить
    return 0.25 * (pos_pos + neg_pos + pos_neg + neg_neg)


def unpack_quarter(f_q, N=None):
    """Разворачивает четверть (N, N//2, N//2) обратно в полный массив (N, N, N),
    копируя данные в четыре симметричные подобласти. Это обратная операция к
    pack_quarter и точно восстанавливает f, если f изначально была симметрична
    по ξ_y, ξ_z."""
    if N is None:
        N = f_q.shape[1] * 2
    h = N // 2
    f = np.empty((f_q.shape[0], N, N))
    f[:, h:, h:] = f_q
    f[:, :h, h:] = f_q[:, ::-1, :]
    f[:, h:, :h] = f_q[:, :, ::-1]
    f[:, :h, :h] = f_q[:, ::-1, ::-1]
    return f


def compute_macro_parameters(f, xi_grid, dxi):
    """Compute macroscopic parameters n, u, T."""
    volume = dxi[0] * dxi[1] * dxi[2]
    X, Y, Z = np.meshgrid(xi_grid[0], xi_grid[1], xi_grid[2], indexing='ij')

    n = np.sum(f) * volume
    if n == 0: return 0.0, 0.0, 0.0

    Vx = np.sum(f * X) * volume / n
    Vy = np.sum(f * Y) * volume / n
    Vz = np.sum(f * Z) * volume / n

    T_xx = np.sum(f * (X - Vx)**2) * volume / n
    T_yy = np.sum(f * (Y - Vy)**2) * volume / n
    T_zz = np.sum(f * (Z - Vz)**2) * volume / n

    T = (T_xx + T_yy + T_zz) / 3.0

    mask = f > 1e-16
    H_val = np.sum(f[mask] * np.log(f[mask])) * volume

    return T, T_xx, H_val
