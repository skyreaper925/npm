# -*- coding: utf-8 -*-
from sem1 import *


def create_velocity_grid(Nx, Ny, Nz, xi_cut):
    """
    1) Создаёт трёхмерную равномерную скоростную сетку, симметричную относительно начала координат,
       без узлов с нулевой компонентой скорости.

    Параметры:
    Nx, Ny, Nz : int
        Число узлов по осям ξx, ξy, ξz (рекомендуется 20).
    xi_cut : float
        Скорость обрезания.

    Возвращает:
    grid : tuple of np.ndarray
        (ξx, ξy, ξz) — одномерные массивы координат сетки.
    dxi : tuple of float
        Шаг по каждой из трёх осей.
    """
    # координаты от –xi_cut+dxi/2 до +xi_cut–dxi/2, чтобы не было нулевых узлов
    dxi = (2 * xi_cut) / Nx, (2 * xi_cut) / Ny, (2 * xi_cut) / Nz
    xi_x = np.linspace(-xi_cut + dxi[0] / 2, xi_cut - dxi[0] / 2, Nx)
    xi_y = np.linspace(-xi_cut + dxi[1] / 2, xi_cut - dxi[1] / 2, Ny)
    xi_z = np.linspace(-xi_cut + dxi[2] / 2, xi_cut - dxi[2] / 2, Nz)
    xi = np.array((xi_x, xi_y, xi_z))
    return xi, dxi


"""
    2)
    из sem_1:
        create_random_params(n: int)
        normalize_params(a: np.ndarray, d: float, xi_cut: float)
"""


def snap_collision_velocities(a, xi_grid, dxi):
    """
    3) «Привязывает» скорости столкновения к ближайшим узлам сетки.

    Параметры:
    a : np.ndarray, shape (8, n)
        Нормированный массив узлов; a[:6] — скорости.
    xi_grid : tuple of np.ndarray
        (ξx, ξy, ξz) — одномерные массивы координат сетки, симметричной относительно начала координат.
    dxi : tuple of float
        Шаги по каждой оси.

    Возвращает:
    a_snapped : np.ndarray, shape (8, n)
        Узлы, у которых скорости заменены на ближайшие сеточные значения.
    snapped_idx : np.ndarray, shape (6, n)
        Координаты ближайших узлов на сетке для двух скоростей.
    """
    xi_x, xi_y, xi_z = xi_grid
    a_snapped = a.copy()
    n = np.shape(a)[1]  # число столкновений
    snapped_idx = np.zeros((6, n))
    # для каждого из 6 компонентов скорости
    for i, grid in enumerate(
            (xi_x, xi_y, xi_z, xi_x, xi_y, xi_z)):  # grid — сетка узлов, соответствующая текущей компоненте
        # индекс ближайшего узла
        idx = np.round((a[i] - grid[0]) / dxi[i % 3]).astype(
            int)  # вычисляем, на сколько шагов каждая скорость a[i] отстоит от начала сетки.
        idx = np.clip(idx, 0, len(grid) - 1)  # ограничивает индексы, чтобы они не выходили за границы массива grid.
        snapped_idx[i] = idx
        a_snapped[i] = grid[idx]
    return a_snapped, snapped_idx


def filter_collision_nodes(a, snapped_idx, xi_cut):
    """
    4) Отсеивает столкновения, у которых модуль полной скорости хотя бы одной частицы превышает xi_cut.

    Параметры:
    a : np.ndarray, shape (8, n)
        Узлы со скоростями, приближенными сеткой. a[0:3] — скорость первой частицы, a[3:6] — второй.
    snapped_idx : np.ndarray, shape (6, n)
        Координаты ближайших узлов на сетке для двух скоростей.
    xi_cut : float
        Предельная длина скорости.

    Возвращает:
    a_filtered : np.ndarray, shape (8, m)
        Узлы, в которых длины обеих скоростей меньше xi_cut (m ≤ n).
    snapped_idx_filtered : np.ndarray, shape (6, m)
        Координаты на сетке узлов, в которых длины обеих скоростей меньше xi_cut (m ≤ n).
    """
    # Считаем длину вектора скорости для каждой частицы (по всем n узлам)
    speed1 = np.sqrt(np.sum(a[0:3] * a[0:3], axis=0))  # длины скоростей ξ
    speed2 = np.sqrt(np.sum(a[3:6] * a[3:6], axis=0))  # длины скоростей ξ₁

    # Логическая маска: обе скорости должны быть меньше xi_cut
    mask = (speed1 < xi_cut) & (speed2 < xi_cut)

    # Возвращаем только те узлы, которые проходят фильтр
    return a[:, mask], snapped_idx[:, mask]


def compute_post_collision_velocities_sem_2(a, d):
    """
    5) Вычисляет скорости после столкновения по формулам (1.3)–(1.8).

    Перед этим отсекаются столкновения с нулевой относительной скоростью.

    Параметры:
    a : np.ndarray, shape (8, n)
        Нормированные узлы после фильтрации:
        первые 6 строк — ξ, ξ₁, 7-я — b (прицельное расстояние), 8-я — ε (азимутальный угол).
    d : float
        Диаметр частиц.

    Возвращает:
    updated_a : np.ndarray, shape (8, m)
        Обновлённые узлы: первые 6 строк — ξ', ξ₁', 7-я — b, 8-я — ε.
    """
    # 3. Вычисление относительных скоростей
    xi, xi1, g = compute_relative_velocities(a)

    # Примечание: удалим столкновения с нулевой относительной скоростью
    g_norm = np.sqrt(np.sum(g * g, axis=1))
    mask = g_norm > 1e-10  # допустимые столкновения (у которых относительная скорость больше нуля)

    # Отфильтрованные данные
    a_filtered = a[:, mask]
    xi = xi[mask]
    xi1 = xi1[mask]
    g = g[mask]

    # 4. Углы отклонения (theta), формула 1.14
    theta = compute_theta(a_filtered[6], d)

    # 5. Преобразование относительных скоростей (формулы 1.12a, 1.12b)
    g_new = transform_g(g, a_filtered[7], theta)

    # 6. Расчёт скоростей после столкновения (формула 1.8)
    xi_prime, xi1_prime = compute_post_collision_velocities(xi, xi1, g_new)

    # 7. Сборка массива (8, m) обратно: ξ', ξ₁', b, ε
    updated_a = np.zeros_like(a_filtered)
    updated_a[:3] = xi_prime.T
    updated_a[3:6] = xi1_prime.T
    updated_a[6] = a_filtered[6]  # считаем что тут ничего не поменялось
    updated_a[7] = a_filtered[7]  # считаем что тут ничего не поменялось

    return updated_a


def find_interpolating_nodes_and_weights(a_updated, xi_grid, dxi, xi_cut):
    """
    6) находим λ_ν, λ_ν + s_ν - координаты приближающих узлов для разлётной скорости ξ', μ_ν, μ_ν - s_ν - координаты
    приближающих узлов для скорости ξ₁', а также r_ν
    Параметры:
    updated_a : np.ndarray, shape (8, m)
        Обновлённые узлы после столкновения: первые 6 строк — ξ', ξ₁', 7-я — b, 8-я — ε.

    Возвращает:
    lam1 lam2       mu1  mu2
    λ_ν, λ_ν + s_ν, μ_ν, μ_ν - s_ν : indices of grid nodes
    r       : np.ndarray, shape (n,)
        Коэффициенты интерполяции.
    """
    # А) ищем ближайшие узлы для  xi_p и xi1_p и отсеиваем те которые не попадают в сферу радиуса xi_cut
    a_updated_snapped, snapped_idx_updated = snap_collision_velocities(a_updated, xi_grid, dxi)

    speed1 = np.sqrt(np.sum(a_updated_snapped[0:3] * a_updated_snapped[0:3], axis=0))  # длины скоростей ξ
    speed2 = np.sqrt(np.sum(a_updated_snapped[3:6] * a_updated_snapped[3:6], axis=0))  # длины скоростей ξ₁
    # Логическая маска: обе скорости должны быть меньше xi_cut
    mask = (speed1 < xi_cut) & (speed2 < xi_cut)

    # Возвращаем только те узлы, которые проходят фильтр
    a_updated_snapped_filtered, snapped_idx_updated_filtered = a_updated_snapped[:, mask], snapped_idx_updated[:, mask]
    a_updated_filtered = a_updated[:, mask]  # то что нужно для получения eta_center
    eta_near = snapped_idx_updated_filtered[0:3, :]  # массив ближайших точек из узлов сетки, shape(3, m) для xi
    velocity_near = a_updated_snapped_filtered[0:3, :]  # массив ближайших координат из узлов сетки, shape(3, m) для  xi
    eta_near1 = snapped_idx_updated_filtered[3:6, :]  # то же для скорости xi1
    velocity_near1 = a_updated_snapped_filtered[3:6, :]  #то же для скорости xi1

    # Б) ищем 8 узлов, которые окружают точную разлётную скорость для каждого столкновения
    m = np.shape(a_updated_snapped_filtered)[1]  # длина массива столкновений, которые мы рассматриваем
    eta_center = np.zeros((3, m), dtype=int)
    for i in range(3):  # i = 0 (x), 1 (y), 2 (z)
        eta_center[i] = np.floor(a_updated_filtered[i] / dxi[i] + 0.5 * len(xi_grid[i]) + 0.5)
    eta_center1 = np.zeros((3, m), dtype=int)
    for i in range(3):  # i = 3 (x), 4 (y), 5 (z)
        eta_center1[i] = np.floor(a_updated_filtered[i + 3] / dxi[i] + 0.5 * len(xi_grid[i]) + 0.5)

    shifts_surround = np.zeros((8, 3, m),
                               dtype=int)  # 8 окружающих точек, для каждой 3 координаты скорости относительно eta_center: i, j, k:
    nodes_surround = np.zeros((8, 3, m),
                              dtype=int)  # 8 окружающих точек, для каждой 3 абсолютных координаты скорости (не относительно eta_center)
    nodes_surround1 = np.zeros((8, 3, m),
                               dtype=int)  # 8 окружающих точек, симметричных тем что в nodes_surround, для каждой 3 абсолютных координаты скорости (не относительно eta_center)
    # все сочетания (i,j,k) ∈ {0,1}³
    shifts = [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (1, 0, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    ]
    for v, (si, sj, sk) in enumerate(shifts):
        nodes_surround[v, 0, :] = eta_center[0] + si
        nodes_surround[v, 1, :] = eta_center[1] + sj
        nodes_surround[v, 2, :] = eta_center[2] + sk
    for v, (si, sj, sk) in enumerate(shifts):  # симметричные 8 точек
        nodes_surround1[v, 0, :] = eta_center1[0] - si
        nodes_surround1[v, 1, :] = eta_center1[1] - sj
        nodes_surround1[v, 2, :] = eta_center1[2] - sk
    for v, (si, sj, sk) in enumerate(shifts):
        shifts_surround[v, 0, :] = si
        shifts_surround[v, 1, :] = sj
        shifts_surround[v, 2, :] = sk

    # ограничиваем индексы, чтобы они не выходили за границы массива grid (теперь среди 8 окружающих точек могут появляться совпадающие)
    nodes_surround[:, 0, :] = np.clip(nodes_surround[:, 0, :], 0, len(xi_grid[0]) - 1)
    nodes_surround[:, 1, :] = np.clip(nodes_surround[:, 1, :], 0, len(xi_grid[1]) - 1)
    nodes_surround[:, 2, :] = np.clip(nodes_surround[:, 2, :], 0, len(xi_grid[2]) - 1)
    nodes_surround1[:, 0, :] = np.clip(nodes_surround1[:, 0, :], 0, len(xi_grid[0]) - 1)
    nodes_surround1[:, 1, :] = np.clip(nodes_surround1[:, 1, :], 0, len(xi_grid[1]) - 1)
    nodes_surround1[:, 2, :] = np.clip(nodes_surround1[:, 2, :], 0, len(xi_grid[2]) - 1)

    velocity_surround = np.zeros((8, 3, m), dtype=float)  # тот же массив, но в нём реальные проекции скоростей
    velocity_surround1 = np.zeros((8, 3, m), dtype=float)
    # преобразуем индексы в реальные скорости
    for v in range(8):  # 8 вершин
        for dim in range(3):  # 0→x,1→y,2→z
            grid = xi_grid[dim]  # выбираем нужную одномерную сетку
            ids = nodes_surround[v, dim, :]  # индексы вдоль этой оси для всех m столкновений
            velocity_surround[v, dim, :] = grid[ids]
    for v in range(8):  # 8 вершин
        for dim in range(3):  # 0→x,1→y,2→z
            grid = xi_grid[dim]  # выбираем нужную одномерную сетку
            ids = nodes_surround1[v, dim, :]  # индексы вдоль этой оси для всех m столкновений
            velocity_surround1[v, dim, :] = grid[ids]

    # e_near = eta_near - eta_center  # координата ближайшего узла относительно центра

    # В) составляются кинетические энергии этих узлов.
    xi_c = 0.5 * (a_updated[0:3, mask] + a_updated[3:6, mask])  # центр масс скоростей, shape(3, m)
    E_surround = np.zeros((8, m))  # энергии соседних столкновений
    for v in range(8):
        E_surround[v, :] = (velocity_surround[v, 0, :] - xi_c[0, :]) ** 2 + (
                velocity_surround[v, 1, :] - xi_c[1, :]) ** 2 + (velocity_surround[v, 2, :] - xi_c[2, :]) ** 2

    # Г) находим E_near - энергию ближайшего узла
    E_near = (velocity_near[0, :] - xi_c[0, :]) ** 2 + (velocity_near[1, :] - xi_c[1, :]) ** 2 + (
            velocity_near[2, :] - xi_c[2, :]) ** 2  # энергия ближайшего узла
    E_0 = (a_updated_filtered[0, :] - xi_c[0, :]) ** 2 + (a_updated_filtered[1, :] - xi_c[1, :]) ** 2 + (
            a_updated_filtered[2, :] - xi_c[2, :]) ** 2  # энергия точной разлётной скорости

    # Д) В соответствии с (2.8) для случая 2 нужно найти такую выборку из соседних узлов, для которых верно E_i,j,k < E0
    # Инициализация массивов
    m = a_updated_filtered.shape[1]
    lam1 = np.zeros((3, m), dtype=int)  # λ_ν (ближайший узел)
    lam2 = np.zeros((3, m), dtype=int)  # λ_ν + s_ν (соседний узел)
    mu1 = np.zeros((3, m), dtype=int)  # μ_ν
    mu2 = np.zeros((3, m), dtype=int)  # μ_ν - s_ν
    r = np.zeros(m)
    keep = np.ones(m, dtype=bool)  # Маска для валидных столкновений

    for i in range(m):
        # Случай 1: E0 == E_near
        if np.isclose(E_0[i], E_near[i]):
            lam1[:, i] = eta_near[:, i]
            lam2[:, i] = eta_near[:, i]
            mu1[:, i] = eta_near1[:, i]
            mu2[:, i] = eta_near1[:, i]
            r[i] = 1.0

        elif E_0[i] < E_near[i]:
            lam2[:, i] = eta_near[:, i]
            mu2[:, i] = eta_near1[:, i]

            candidates = np.where(E_surround[:, i] < E_0[i])[0]  # массив индексов
            # duplicate start
            if candidates.size == 0:
                keep[i] = False
                continue

            candidates_velocity = velocity_surround[candidates, :, i]
            candidates_velocity1 = velocity_surround1[candidates, :, i]
            candidate_nodes_velocity = nodes_surround[candidates, :, i]
            candidate_nodes_velocity1 = nodes_surround1[candidates, :, i]

            # не берем те узлы, в которых скорости выходят за сферу радиуса xi_cut
            valid_mask = ((np.sqrt(np.sum(candidates_velocity * candidates_velocity, axis=1)) <= xi_cut) &
                          (np.sqrt(np.sum(candidates_velocity1 * candidates_velocity1, axis=1)) <= xi_cut))
            if not np.any(valid_mask):
                keep[i] = False
                continue

            candidates_velocity_filtered = candidates_velocity[valid_mask, :]
            # candidates_velocity_filtered1 = candidates_velocity1[valid_mask, :]
            candidate_nodes_velocity_filtered = candidate_nodes_velocity[valid_mask, :]
            candidate_nodes_velocity_filtered1 = candidate_nodes_velocity1[valid_mask, :]
            if len(candidates_velocity_filtered) == 0:
                keep[i] = False
                continue

            # Находим ближайший узел к разлётной скорости
            tmp = candidates_velocity_filtered[:, :] - a_updated_filtered[:3, i]
            dist = np.sqrt(np.sum(tmp * tmp, axis=1))
            chosen_idx = np.argmin(dist)
            # duplicate end

            lam1[:, i] = candidate_nodes_velocity_filtered[chosen_idx]
            mu1[:, i] = candidate_nodes_velocity_filtered1[chosen_idx]

            # Коэффициент r из условия (2.4)
            E2 = E_near[i]
            E1 = (candidates_velocity_filtered[chosen_idx, 0] - xi_c[0, i]) ** 2 + (
                    candidates_velocity_filtered[chosen_idx, 1] - xi_c[1, i]) ** 2 + (
                         candidates_velocity_filtered[chosen_idx, 2] - xi_c[2, i]) ** 2
            r[i] = (E_0[i] - E1) / (E2 - E1)

        else:
            lam1[:, i] = eta_near[:, i]
            mu1[:, i] = eta_near1[:, i]
            candidates = np.where(E_surround[:, i] > E_0[i])[0]  # массив индексов
            # duplicate start
            if candidates.size == 0:
                keep[i] = False
                continue

            candidates_velocity = velocity_surround[candidates, :, i]
            candidates_velocity1 = velocity_surround1[candidates, :, i]
            candidate_nodes_velocity = nodes_surround[candidates, :, i]
            candidate_nodes_velocity1 = nodes_surround1[candidates, :, i]

            # не берем те узлы, в которых скорости выходят за сферу радиуса xi_cut
            valid_mask = ((np.sqrt(np.sum(candidates_velocity * candidates_velocity, axis=1)) <= xi_cut) &
                          (np.sqrt(np.sum(candidates_velocity1 * candidates_velocity1, axis=1)) <= xi_cut))
            if not np.any(valid_mask):
                keep[i] = False
                continue

            candidates_velocity_filtered = candidates_velocity[valid_mask, :]
            # candidates_velocity_filtered1 = candidates_velocity1[valid_mask, :]
            candidate_nodes_velocity_filtered = candidate_nodes_velocity[valid_mask, :]
            candidate_nodes_velocity_filtered1 = candidate_nodes_velocity1[valid_mask, :]
            if len(candidates_velocity_filtered) == 0:
                keep[i] = False
                continue

            # Находим ближайший узел к разлётной скорости
            tmp = candidates_velocity_filtered[:, :] - a_updated_filtered[:3, i]
            dist = np.sqrt(np.sum(tmp * tmp, axis=1))
            chosen_idx = np.argmin(dist)
            # duplicate end
            lam2[:, i] = candidate_nodes_velocity_filtered[chosen_idx]
            mu2[:, i] = candidate_nodes_velocity_filtered1[chosen_idx]

            # Коэффициент r из условия (2.4)
            E1 = E_near[i]
            E2 = (candidates_velocity_filtered[chosen_idx, 0] - xi_c[0, i]) ** 2 + (
                    candidates_velocity_filtered[chosen_idx, 1] - xi_c[1, i]) ** 2 + (
                         candidates_velocity_filtered[chosen_idx, 2] - xi_c[2, i]) ** 2
            r[i] = (E_0[i] - E1) / (E2 - E1)

    lam1 = lam1[:, keep]
    lam2 = lam2[:, keep]
    mu1 = mu1[:, keep]
    mu2 = mu2[:, keep]
    r = r[keep]

    return lam1, lam2, mu1, mu2, r
