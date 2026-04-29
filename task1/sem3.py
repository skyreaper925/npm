import random
from math import sqrt, pi
from sem2 import *
from sem1 import *


def eratosthenes_sieve(n):
    """Generate prime numbers up to n using Sieve of Eratosthenes."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(sqrt(n)) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(n + 1) if sieve[i]]


def H(z, p, s=5):
    """Compute H(z) function for Korobov grid coefficients (Eq. 3.35)."""
    total = 0.0
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
    H_values = []
    for z in range(1, (p - 1) // 2 + 1):
        H_values.append((H(z, p), z))
    min_H = min(H_values, key=lambda x: x[0])[0]
    b_candidates = [h[1] for h in H_values if h[0] == min_H]
    b = b_candidates[0]  # Choose first candidate
    a = [1]
    current = b
    for _ in range(1, p):
        a.append(current)
        current = (current * b) % p
    return a


def initialize_distribution(N, xi_cut):
    xi, dxi = create_velocity_grid(N, N, N, xi_cut)
    # Создаем полноценную 3D сетку
    X, Y, Z = np.meshgrid(xi[0], xi[1], xi[2], indexing='ij')
    xi_sq = X**2 + Y**2 + Z**2

    T_c, T_h = 1.0, 2.0
    f_c = np.exp(-0.5 * xi_sq / T_c) / (2 * pi * T_c)**1.5
    f_h = np.exp(-0.5 * xi_sq / T_h) / (2 * pi * T_h)**1.5

    volume = dxi[0] * dxi[1] * dxi[2]
    # Нормируем так, чтобы n_cold = 0.5, n_hot = 0.5, n_total = 1.0
    f_c = 0.5 * f_c / (np.sum(f_c) * volume)
    f_h = 0.5 * f_h / (np.sum(f_h) * volume)

    return f_c, f_h, xi, dxi

def find_interpolation_nodes(xi_prime, xi_1_prime, xi_grid, delta_xi, xi_cut):
    N = len(xi_grid[0])

    def get_nearest_idx(xi_val, grid_1d, dx):
        # Находим два соседних сеточных узла, окружающих точку
        idx = int(np.floor((xi_val - grid_1d[0]) / dx + 0.5))
        return max(0, min(N - 1, idx))

    lambda_idx = [get_nearest_idx(xi_prime[i], xi_grid[i], delta_xi[i]) for i in range(3)]
    mu_idx = [get_nearest_idx(xi_1_prime[i], xi_grid[i], delta_xi[i]) for i in range(3)]

    # Строго по Пособию: s_i = sign(xi'_i - xi_lambda)
    s_v = [1 if xi_prime[i] >= xi_grid[i][lambda_idx[i]] else -1 for i in range(3)]
    p_v = [1 if xi_1_prime[i] >= xi_grid[i][mu_idx[i]] else -1 for i in range(3)]

    # Индексы сдвинутых узлов (с программной защитой от отрицательных индексов Python)
    lambda_s_idx = [max(0, min(N - 1, lambda_idx[i] + s_v[i])) for i in range(3)]
    mu_p_idx = [max(0, min(N - 1, mu_idx[i] + p_v[i])) for i in range(3)]

    E0 = sum(xi_prime[i]**2 for i in range(3)) + sum(xi_1_prime[i]**2 for i in range(3))
    E1 = sum(xi_grid[i][lambda_idx[i]]**2 for i in range(3)) + sum(xi_grid[i][mu_idx[i]]**2 for i in range(3))
    E2 = sum(xi_grid[i][lambda_s_idx[i]]**2 for i in range(3)) + sum(xi_grid[i][mu_p_idx[i]]**2 for i in range(3))

    # Расчет коэффициента r_v с жестким ограничением [0, 1] для гарантии стабильности
    if E2 != E1:
        r_v = (E0 - E1) / (E2 - E1)
        r_v = max(0.0, min(1.0, r_v))
    else:
        r_v = 0.5

    # Защита от накопления машинной погрешности (должно быть в пределах [0, 1])
    r_v = max(0.0, min(1.0, r_v))

    return lambda_idx, mu_idx, lambda_s_idx, mu_p_idx, r_v


def compute_collision_integral(f_c, f_h, xi_grid, dxi, xi_cut, N_v, b_max=1.0, tau=0.02):
    N = f_c.shape[0]
    volume = dxi[0] * dxi[1] * dxi[2]
    V_sph = 4 * pi * xi_cut ** 3 / 3
    N_0 = round(V_sph / volume)
    C = (b_max ** 2 * volume * N_0 ** 2 / N_v * tau) / (2 ** 2.5)

    p = int(4 * N_v)
    primes = eratosthenes_sieve(p + 100)
    candidates = [x for x in primes if x >= p]
    if candidates: p = min(candidates)
    a = find_korobov_coefficients(p)

    grid = np.array([((i * a[j]) % p) / p for j in range(5) for i in range(1, p + 1)]).reshape(-1, 5)
    grid[:, 0] *= 2 * pi
    grid[:, 1] *= b_max ** 2
    grid[:, 2:5] = grid[:, 2:5] * 2 * xi_cut - xi_cut
    random.shuffle(grid)

    f_c_new, f_h_new = f_c.copy(), f_h.copy()
    f_tot = f_c + f_h
    
    for v in range(min(N_v, len(grid))):
        epsilon, S_b, xi_1_continuous = grid[v, 0], grid[v, 1], grid[v, 2:5]
        if np.linalg.norm(xi_1_continuous) > xi_cut: continue

        # ИСПРАВЛЕНИЕ: привязка xi_1 к сетке ДО расчета столкновения
        beta_idx = tuple(max(0, min(N - 1, int(np.floor((xi_1_continuous[i] - xi_grid[i][0]) / dxi[i] + 0.5)))) for i in range(3))
        xi_beta = np.array([xi_grid[i][beta_idx[i]] for i in range(3)])

        alpha_idx = tuple(random.randint(0, N - 1) for _ in range(3))
        xi_alpha = np.array([xi_grid[i][alpha_idx[i]] for i in range(3)])

        # Теперь кинематика рассчитывается строго на основе сеточных энергий
        g = xi_beta - xi_alpha
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-10: continue

        # Реальное столкновение на основе узла \alpha и точки xi_1 с сетки
        b = np.sqrt(S_b)
        theta = compute_theta(np.array([b]), 1.0)[0]
        g_new = transform_g(np.array([g]), np.array([epsilon]), np.array([theta]))[0]

        xi_prime = 0.5 * (xi_alpha + xi_beta) - 0.5 * g_new
        xi_1_prime = 0.5 * (xi_alpha + xi_beta) + 0.5 * g_new

        if np.linalg.norm(xi_prime) > xi_cut or np.linalg.norm(xi_1_prime) > xi_cut:
            continue

        lambda_idx, mu_idx, lambda_s_idx, mu_p_idx, r_v = find_interpolation_nodes(
            xi_prime, xi_1_prime, xi_grid, dxi, xi_cut)

        lambda_idx, mu_idx = tuple(lambda_idx), tuple(mu_idx)
        lambda_s_idx, mu_p_idx = tuple(lambda_s_idx), tuple(mu_p_idx)

        def get_vals(dist):
            return (dist[alpha_idx], dist[beta_idx],
                    dist[lambda_idx], dist[mu_idx],
                    dist[lambda_s_idx], dist[mu_p_idx])

        fc_a, fc_b, fc_l, fc_m, fc_ls, fc_mp = get_vals(f_c)
        fh_a, fh_b, fh_l, fh_m, fh_ls, fh_mp = get_vals(f_h)
        ftot_a, ftot_b, ftot_l, ftot_m, ftot_ls, ftot_mp = get_vals(f_tot)

        # Значения функции распределения в точках разлета (аппроксимация по Пособию).
        # Используем линейную формулу (1-r)*f_l + r*f_ls, так как для смесей
        # геометрическое среднее даст ноль в случае пустых зон f_c или f_h.
        fc_prime = (1 - r_v) * fc_l + r_v * fc_ls
        fc_1_prime = (1 - r_v) * fc_m + r_v * fc_mp

        fh_prime = (1 - r_v) * fh_l + r_v * fh_ls
        fh_1_prime = (1 - r_v) * fh_m + r_v * fh_mp

        ftot_prime = (1 - r_v) * ftot_l + r_v * ftot_ls
        ftot_1_prime = (1 - r_v) * ftot_m + r_v * ftot_mp

        # Вычисление приращений.
        # Поскольку частицы структурно одинаковы, частота столкновений определяется полной концентрацией f_tot.
        Omega_c = (fc_prime * ftot_1_prime - fc_a * ftot_b) * g_norm
        Omega_c_sym = (ftot_prime * fc_1_prime - ftot_a * fc_b) * g_norm

        Omega_h = (fh_prime * ftot_1_prime - fh_a * ftot_b) * g_norm
        Omega_h_sym = (ftot_prime * fh_1_prime - ftot_a * fh_b) * g_norm

        def apply_delta(delta_f, O_1, O_2):
            delta_f[alpha_idx] += C * O_1
            delta_f[beta_idx] += C * O_2
            delta_f[lambda_idx] -= (1 - r_v) * C * O_1
            delta_f[lambda_s_idx] -= r_v * C * O_1
            delta_f[mu_idx] -= (1 - r_v) * C * O_2
            delta_f[mu_p_idx] -= r_v * C * O_2

        delta_f_c, delta_f_h = np.zeros_like(f_c), np.zeros_like(f_h)
        apply_delta(delta_f_c, Omega_c, Omega_c_sym)
        apply_delta(delta_f_h, Omega_h, Omega_h_sym)

        f_c_temp, f_h_temp = f_c_new + delta_f_c, f_h_new + delta_f_h
        if np.any(f_c_temp < 0) or np.any(f_h_temp < 0): continue

        f_c_new, f_h_new = f_c_temp, f_h_temp

    return f_c_new, f_h_new


def compute_macro_parameters(f, xi_grid, dxi):
    """Compute macroscopic parameters n, u, T."""
    volume = dxi[0] * dxi[1] * dxi[2]
    n = np.sum(f) * volume
    X, Y, Z = np.meshgrid(xi_grid[0], xi_grid[1], xi_grid[2], indexing='ij')

    u_x = np.sum(f * X) / np.sum(f)
    u_y = np.sum(f * Y) / np.sum(f)
    u_z = np.sum(f * Z) / np.sum(f)

    T = np.sum(f * ((X - u_x)**2 + (Y - u_y)**2 + (Z - u_z)**2)) / (3 * np.sum(f))
    S = -np.sum(f * np.log(f + 1e-16)) * volume

    return n, T, S


def newton_method(xi_grid, dxi, u_guess, T_guess):
    """Find u*, T* for Maxwellian distribution using Newton's method."""

    def compute_y(u_star, T_star):
        volume = dxi[0] * dxi[1] * dxi[2]
        f_M = np.exp(-0.5 * np.sum((xi_grid - u_star) ** 2, axis=0) / T_star)
        norm = np.sum(f_M) * volume
        f_M /= norm
        xi_mean = np.sum(f_M * xi_grid) * volume
        xi_sq_mean = np.sum(f_M * np.sum(np.array(xi_grid) ** 2, axis=0) * volume)
        y1 = (xi_sq_mean - u_star[0] ** 2) / 3 - T_guess
        y2 = xi_mean - u_guess[0]
        return y1, y2

    N = xi_grid.shape[1]
    u_star = np.zeros((3, N))
    u_star[0][0] = u_guess[0]
    T_star = T_guess
    for _ in range(10):
        y1, y2 = compute_y(u_star, T_star)
        if abs(y1).all() < 1e-6 and abs(y2) < 1e-6:
            break
        # Compute partial derivatives numerically
        eps = 1e-6
        y1_u, y2_u = compute_y(u_star[0] + eps, T_star)
        y1_T, y2_T = compute_y(u_star, T_star + eps)
        dy1_du = (y1_u - y1) / eps
        dy1_dT = (y1_T - y1) / eps
        dy2_du = (y2_u - y2) / eps
        dy2_dT = (y2_T - y2) / eps
        J = np.array([[dy1_dT[0], dy1_du[0]], [dy2_dT, dy2_du]])
        delta = np.linalg.solve(J, [-y1[0], -y2])
        T_star += delta[0]
        u_star[0] += delta[1]
    return u_star, T_star


def compute_norms(f, f_M, xi_grid, dxi, xi_cut):
    """Compute differentiated norms (Eq. 3.38)."""
    norms = []
    volume = dxi[0] * dxi[1] * dxi[2]
    for k in range(1, 4):
        mask = ((k - 1) * xi_cut / 3 <= np.sqrt(xi_grid[k-1] ** 2)) & \
               (np.sqrt(xi_grid[k-1] ** 2) <= k * xi_cut / 3)

        #print(f[k-1] - f_M[k-1], "\n")
        norm = np.sqrt((f[k-1] - f_M[k-1]) ** 2 * mask) * volume
        norms.append(norm)
    return norms