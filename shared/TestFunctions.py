import numpy as np



def sphere(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    return np.sum(x**2)


def schwefel(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    d = x.size
    return 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def rosenbrock(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def rastrigin(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    d = x.size
    return 10.0 * d + np.sum(x**2 - 10.0 * np.cos(2 * np.pi * x))


def griewank(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    d = x.size
    sum_term = np.sum(x**2) / 4000.0
    i = np.arange(1, d + 1, dtype=float)
    prod_term = np.prod(np.cos(x / np.sqrt(i)))
    return sum_term - prod_term + 1.0


def levy(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    w = 1 + (x - 1) / 4.0
    term1 = np.sin(np.pi * w[0])**2
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2)) if x.size > 1 else 0.0
    return term1 + term2 + term3


def michalewicz(x: np.ndarray, m: float = 10.0):
    x = np.asarray(x, dtype=float)
    i = np.arange(1, x.size + 1, dtype=float)
    return -np.sum(np.sin(x) * (np.sin(i * x**2 / np.pi) ** (2 * m)))


def zakharov(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    i = np.arange(1, x.size + 1, dtype=float)
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * i * x)
    return sum1 + sum2**2 + sum2**4


def ackley(x: np.ndarray, a: float = 20.0, b: float = 0.2, c: float = 2*np.pi):
    x = np.asarray(x, dtype=float)
    d = x.size
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / d))
    term2 = -np.exp(sum_cos / d)
    return term1 + term2 + a + np.e



TEST_FUNCTIONS = {
    sphere: (-5.12, 5.12, 100),
    ackley: (-32.768, 32.768, 100),
    rastrigin: (-5.12, 5.12, 100),
    rosenbrock: (-10, 10, 100),
    griewank: (-50, 50, 100),
    schwefel: (-500, 500, 100),
    levy: (-10, 10, 100),
    michalewicz: (0, np.pi, 100),
    zakharov: (-10, 10, 100)
}
