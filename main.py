import numpy as np
import matplotlib.pyplot as plt
import math
from math import sin, cos
from scipy import constants
from scipy.linalg import eigh
from scipy.integrate import quad
from py_linq import Enumerable

# Параметры
hbar = 1.0545718e-34  # (Дж·с) приведённая постоянная Планка
eV = 1.60218e-19  # (Дж) 1 электрон-вольт
m = 9.10938356e-31  # (кг) масса электрона
a = 1e-9  # [м] ширина ямы (на самом деле «полуширина» в старом примере)
U0 = 5 * eV  # [Дж] глубина ямы (например, 5 эВ)
L = 3 * a  # [м] область, в которой мы будем моделировать
N = 1000  # число точек сетки

x = np.linspace(-L / 2, L / 2, N)  # координатная сетка
dx = x[1] - x[0]

y = []
for xi in x:
    if -a / 2 <= xi <= a / 2:
        y.append(-U0)
    else:
        y.append(0)
V = np.array(y)

plt.figure(figsize=(8, 5))
plt.title("Прямоугольная потенциальная яма")
plt.xlabel("x, м")
plt.ylabel("V(x), эВ")
plt.plot(x, V / eV, color='black', label='V(x)')
plt.grid(True)
plt.legend()
plt.show()

diag_kin = (hbar ** 2) / (2 * m * dx ** 2)

H = (
        np.diag(2 * diag_kin + V) +
        np.diag(-diag_kin * np.ones(N - 1), k=1) +
        np.diag(-diag_kin * np.ones(N - 1), k=-1)
)

eigvals, eigvecs = eigh(H)
energies = eigvals / eV
eigvecs = eigvecs / np.sqrt(dx)

bound_mask = (energies < 0)
bound_energies = energies[bound_mask]
bound_states = eigvecs[:, bound_mask]

n_show = min(len(bound_energies), 3)
plt.figure(figsize=(8, 5))
plt.title("Связанные состояния в прямоугольной потенциальной яме")
plt.xlabel("x, м")
plt.ylabel("Energy, эВ (с волновыми функциями)")

for i in range(n_show):
    psi_i = bound_states[:, i]
    E_i = bound_energies[i]
    psi_plot = psi_i / np.max(np.abs(psi_i)) * 0.4 + E_i
    plt.plot(x, psi_plot, label=f"E_{i + 1} = {E_i:.3f} эВ")
    plt.axhline(E_i, color='black', linestyle='--', linewidth=0.8)

plt.plot(x, V / eV, 'k', label='V(x)')
plt.legend()
plt.grid(True)
plt.show()


eps = 100000
k_2max = math.sqrt(2 * m * U0) / hbar
n_max = math.ceil(k_2max * a / constants.pi)
print(f"Для текущей ямы: k_2max={k_2max:.3g} 1/м, число n_max={n_max}")


leftx = np.arange(0, k_2max, k_2max / eps)
lefty = leftx * a


def intersect(x1, x2, y1, y2, tol):
    """
    Ищем среди точек x1,y1 и x2,y2 такие пары, где расстояние < tol.
    Возвращаем одну "минимальную" точку.
    """
    points = []
    for i in range(min(len(x1), len(x2))):
        dist = np.sqrt((x1[i] - x2[i]) ** 2 + (y1[i] - y2[i]) ** 2)
        if dist <= tol:
            points.append((x1[i], y1[i], dist))
    if not points:
        return None
    points = Enumerable(points)
    min_dist = points.select(lambda x: x[2]).min()
    return points.where(lambda x: x[2] == min_dist).first()[:2]


plt.figure(figsize=(8, 5))
plt.title("Пример поиска пересечений (черновик)")
plt.plot(leftx, lefty, label="left curve", color='red')

intsec = []
for n in range(1, n_max + 1):
    rightx = []
    righty = []
    kgrid = np.arange(0, k_2max, k_2max / eps)
    for j in kgrid:
        val = constants.pi * n - 2 * math.asin((hbar * j) / math.sqrt(2 * m * U0))
        rightx.append(j)
        righty.append(val)
    rightx = np.array(rightx)
    righty = np.array(righty)
    plt.plot(rightx, righty, color='green', alpha=0.4)

    pt = intersect(leftx, rightx, lefty, righty, 0.1)
    if pt is not None:
        plt.plot(pt[0], pt[1], 'o', color='blue')
        intsec.append(pt)

plt.xlabel("k")
plt.ylabel("Some function(k)")
plt.legend()
plt.grid(True)
plt.show()


def U_barrier(x):
    """
    Пример барьера: U = 5e-20 Дж на отрезке [0, 1e-9], иначе 0.
    """
    U0_local = 5e-20
    sigma = 1e-9
    return U0_local * np.where((x >= 0) & (x <= sigma), 1, 0)


E_test = 4.2e-20
m_local = 9.1e-31


def k1(E_):
    return np.sqrt(2 * m_local * E_) / hbar


def k2(E_):
    U0_local = 5e-20
    return np.sqrt(2 * m_local * (E_ - U0_local)) / hbar


def k_func(x_):
    """
    Локальный волновой вектор, если U(x) > E => затухание, если U(x) < E => бегущая волна
    """
    return np.sqrt(2 * m_local * (U_barrier(x_) - E_test)) / hbar if U_barrier(x_) > E_test else 0


x_tunnel = np.linspace(-1e-9, 2e-9, 1000)
Uvals = U_barrier(x_tunnel)

if E_test > 5e-20:
    # энергия частицы выше барьера => частичное прохождение
    k1_val = k1(E_test)
    k2_val = k2(E_test)
    R = ((k1_val - k2_val) / (k1_val + k2_val)) ** 2
    D = 1 - R
else:
    # энергия меньше барьера => возможен туннельный эффект
    # найдём, где U(x) > E
    x_above = (Uvals > E_test)
    if not np.any(x_above):
        D = 1.0
        R = 0.0
    else:
        # найдём границы барьера, где U(x) становится > E
        x_idxs = np.where(x_above)[0]
        i1 = x_idxs[0]
        i2 = x_idxs[-1]
        x1 = x_tunnel[i1]
        x2 = x_tunnel[i2]
        integral, _ = quad(k_func, x1, x2)
        D = np.exp(-2 * integral)
        R = 1 - D

print(f"\n--- Туннелирование через потенциальный барьер ---")
print(f"Энергия E = {E_test:.2e} Дж, барьер ~5e-20 Дж")
print(f"Коэффициент прохождения D = {D:.4f}")
print(f"Коэффициент отражения  R = {R:.4f}")

plt.figure(figsize=(8, 5))
plt.title("Туннелирование через барьер")
plt.plot(x_tunnel, Uvals, color="blue", label="U(x)")
plt.axhline(E_test, color="red", linestyle="--", label="E частицы")
plt.xlabel("x, м")
plt.ylabel("U(x), Дж")
plt.legend()
plt.grid(True)
plt.show()


state_free = 0
state_wall = 1


def solve_equation(alpha, p):
    """
    cos(alpha) + p*sin(alpha)/alpha
    Если p=0 -> просто cos(alpha). Если p очень большое -> 'стенки'
    """
    if alpha == 0:
        return np.inf
    return cos(alpha) + p * sin(alpha) / alpha


class DefaultCase:
    """
    f(alpha) = cos(alpha)+p*sin(alpha)/alpha
    заштриховка области между -1 и 1 (разрешённая зона), если p==state_wall.
    """

    def __init__(self, p, delta=10):
        self.p = p
        self.delta = delta
        self.x = np.arange(-delta, delta, 0.001)
        self.func = []
        self.down = []
        self.up = []

    def show_plot(self):
        self.func.clear()
        self.down.clear()
        self.up.clear()

        for val in self.x:
            self.func.append(solve_equation(val, self.p))
            self.down.append(-1)
            self.up.append(1)

        self.func = np.array(self.func)
        self.down = np.array(self.down)
        self.up = np.array(self.up)

        plt.figure(figsize=(8, 5))
        if self.p == state_free:
            title_txt = "Случай свободного электрона (p=0)"
        else:
            title_txt = "Случай непрозрачных стенок (p=1)"

        plt.title(title_txt)
        plt.plot(self.x, self.func, label="Функция")
        plt.plot(self.x, self.down, 'r--', label="-1")
        plt.plot(self.x, self.up, 'g--', label="1")

        if self.p == state_wall:
            # заштрихуем участок, где -1 <= f <= 1
            allowed_zones = (self.func >= -1) & (self.func <= 1)
            plt.fill_between(self.x, self.down, self.up,
                             where=allowed_zones, color="lightgreen", alpha=0.5,
                             label="Разрешённые зоны")

        plt.xlabel("alpha")
        plt.ylabel("cos(alpha) + p*sin(alpha)/alpha")
        plt.legend()
        plt.grid()
        plt.show()



case_free = DefaultCase(state_free)
case_free.show_plot()

case_wall = DefaultCase(state_wall)
case_wall.show_plot()


m_e = 9.11e-31

def compute_kronig_penney_solution(a_kp, b_kp, potential_height, energy_range):
    """
    a_kp, b_kp: ширины участков (скажем, a_kp — ширина "ямы",
                b_kp — ширина барьера)
    potential_height: высота барьера (в эВ), относительно дна ямы.
    energy_range: максимум энергии (в эВ), до которого смотрим.
    """
    alpha_0 = np.sqrt(2 * m_e * potential_height * eV / hbar ** 2)  # 1/м
    a0 = a_kp * 1e-10
    b0 = b_kp * 1e-10

    def kp_below_one(En):
        return ((1 - 2 * En) / (2 * np.sqrt(En * (1 - En)))) * \
            np.sin(alpha_0 * a0 * np.sqrt(En)) * \
            np.sinh(alpha_0 * b0 * np.sqrt(1 - En)) + \
            np.cos(alpha_0 * a0 * np.sqrt(En)) * \
            np.cosh(alpha_0 * b0 * np.sqrt(1 - En))

    def kp_above_one(En):
        return ((1 - 2 * En) / (2 * np.sqrt(En * (En - 1)))) * \
            np.sin(alpha_0 * a0 * np.sqrt(En)) * \
            np.sin(alpha_0 * b0 * np.sqrt(En - 1)) + \
            np.cos(alpha_0 * a0 * np.sqrt(En)) * \
            np.cos(alpha_0 * b0 * np.sqrt(En - 1))

    E_vals = np.linspace(1e-4, energy_range, 5000)
    # piecewise: ниже порога (En<1) -> kp_below_one, выше (En>1) -> kp_above_one
    f_vals = np.piecewise(E_vals, [E_vals < 1, E_vals > 1],
                          [kp_below_one, kp_above_one, 0])

    return E_vals, f_vals


def calculate_energy_bands(E_vals, f_vals, a_kp, b_kp, extend=1):
    """
    Ищем области, где |f_vals| <= 1, т.к. это означает, что cos(k*(a+b)) = f_vals
    => k*(a+b) = arccos( f_vals ), если -1 <= f_vals <=1
    """
    a_tot = (a_kp + b_kp) * 1e-10
    k_sets = []
    E_sets = []

    # Собираем «полосы» (где функция лежит внутри [-1; +1])
    inside = (f_vals >= -1) & (f_vals <= 1)

    cluster = []
    clusterE = []
    for i in range(len(E_vals)):
        if inside[i]:
            cluster.append(f_vals[i])
            clusterE.append(E_vals[i])
        else:
            if cluster:
                k_sets.append(np.arccos(cluster))
                E_sets.append(clusterE)
                cluster = []
                clusterE = []
    if cluster:
        k_sets.append(np.arccos(cluster))
        E_sets.append(clusterE)


    extended_k = []
    extended_E = []
    for i, karr in enumerate(k_sets):
        kreal = karr / a_tot
        eband = np.array(E_sets[i])

        bigK = []
        bigE = []
        for n in range(-extend, extend + 1):
            shift = 2 * math.pi * n / a_tot
            bigK.append(kreal + shift)
            bigE.append(eband)
        bigK = np.concatenate(bigK)
        bigE = np.concatenate(bigE)

        extended_k.append(bigK)
        extended_E.append(bigE)

    return extended_k, extended_E


a_kp = 5
b_kp = 5
potential_height_eV = 1
energy_range_eV = 6

E_vals, F_vals = compute_kronig_penney_solution(a_kp, b_kp, potential_height_eV, energy_range_eV)
k_arrays, E_arrays = calculate_energy_bands(E_vals, F_vals, a_kp, b_kp, extend=2)

plt.figure(figsize=(8, 5))
plt.title("Модель Кронига–Пенни (энергетические зоны)")
for kvals, evals in zip(k_arrays, E_arrays):
    plt.plot(kvals, evals, 'k', linewidth=0.8)

plt.xlabel("k (1/м, условно)")
plt.ylabel("E (эВ)")
plt.grid(True)
plt.show()

