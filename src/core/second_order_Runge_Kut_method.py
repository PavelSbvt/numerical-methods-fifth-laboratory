import numpy as np
from numpy.typing import NDArray

# вторая цифра варианта: h0=0.5, x0 = 1, b = 6, y0 = 10, g(x) = 2*(x - 2)
# третье число варианта: фи(x) = sin(x + 1)
# четвёртое число варианта: функция(x) = 1/8

x0 = 1.0
b = 6.0
y0 = 10.0
h0 = 0.5
eps = 1e-3

def f(x: float, y: float) -> float:
    """Функция для вычисления значения функции f(x)"""
    return (1/8)*np.sin(x + 1) - 2*(x - 2)*y


def runge_kute_2_step(x: float, y: float, h: float) -> float:
    """Функция вычисления средней точки для метода Рунге-Кутта 2-го порядка"""
    k1 = f(x, y)
    k2 = f(x + h/2.0, y + h/2.0 * k1)
    return y + h*k2


def runge_kute_step_with_double(x: float, y: float, h: float) -> tuple[float, float]:
    """
    Функция для вычисления одного шага с удвоением
    :return: возвращет y на конце шага h и y на конце шага h/2 (сделав два шага по h/2)
    """
    y_h = runge_kute_2_step(x, y, h)

    y_h2 = runge_kute_2_step(x, y, h/2.0)

    y_h2 = runge_kute_2_step(x + h/2.0, y_h2, h/2.0)
    return y_h, y_h2


def solve_auto_step(x0: float,
                    b: float,
                    y0: float,
                    h0: float,
                    eps) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """
    Функция для решения дифференциального уравнения методом Рунге-Кутты второго
    порядка с автоматическим шагом.

    :param eps: точность, указанная в научной нотации

    :return: кортеж из двух массивов с координатами x и y и значение шага h
    """
    x = x0
    y = y0
    h = h0
    xs_auto = [x]
    ys_auto = [y]

    while x < b:
        if (x + h) > b:
            h = b -x

        y_h, y_h2 = runge_kute_step_with_double(x, y, h)

        # Оценка погрешности по правилу Рунге
        # Для метода 2-го порядка (p=2), делим на (2^2 - 1) = 3
        error = abs(y_h2 - y_h) / 3.0

        if error <= eps:
            y = y_h2 + (y_h2 - y_h) / 3.0 # Уточнение Ричардсона - повышает точность

            x += h

            xs_auto.append(x)
            ys_auto.append(y)

            if error < eps / 4.0:
                h *= 2.0
                if h > h0:
                    h = h0
        else:
            h /= 2.0

    return np.array(xs_auto), np.array(ys_auto), h


def solve_with_fixed_step(x0: float,
                          b: float,
                          y0: float,
                          N: int
                          ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """
    Функция для вычисления дифференциального уравнения с фиксированным шагом
    :return: массивы координат + шаг
    """
    h = (b - x0) / N
    xs = np.linspace(x0, b, N + 1)
    ys = np.zeros(N + 1)
    ys[0] = y0
    for i in range(N):
        ys[i+1] = runge_kute_2_step(xs[i], ys[i], h)
    return xs, ys, h
