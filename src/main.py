# Метод Рунге-Кута второго порядка
# вторая цифра варианта: h0=0.5, x0 = 1, b = 6, y0 = 10, g(x) = 2*(x - 2)
# третье число варианта: фи(x) = sin(x + 1)
# четвёртое число варианта: функция(x) = 1/8
# Вариант 3, 1, 10, 11
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from core.second_order_Runge_Kut_method import (
    solve_with_fixed_step,
    solve_auto_step,
    f,
    x0, b, y0, h0, eps)
from utils.output_rich import Rich


def print_steps_table(xs_auto: NDArray[np.float64], ys_auto: NDArray[np.float64]):
    cols = ["Шаг i", "x_i", "y_i", "Размер шага (h)"]

    rows = []
    for i in range(len(xs_auto)):
        if i < len(xs_auto) - 1:
            # Вычисляем шаг
            h_current = xs_auto[i + 1] - xs_auto[i]
            h_str = f"{h_current:.4f}"
        else:
            h_str = "-"

        rows.append([
            i,
            f"{xs_auto[i]:.4f}",
            f"{ys_auto[i]:.4f}",
            h_str
        ])

    Rich.print_table(
        title="Таблица шагов (Автоматический подбор)",
        cols=cols,
        rows=rows
    )


def print_fixed_steps_table(xs_fixed, ys_fixed):
    """
    Вывод таблицы шагов для решения с постоянным шагом.
    """
    cols = ["Шаг i", "x_i", "y_i", "Размер шага (h)"]
    rows = []

    for i in range(len(xs_fixed)):
        if i < len(xs_fixed) - 1:
            h_current = xs_fixed[i + 1] - xs_fixed[i]
            h_str = f"{h_current:.6f}"
        else:
            h_str = "-"

        rows.append([
            i,
            f"{xs_fixed[i]:.4f}",
            f"{ys_fixed[i]:.4f}",
            h_str
        ])

    Rich.print_table(
        title="Таблица шагов (Постоянный шаг)",
        cols=cols,
        rows=rows
    )


def print_comparison_table(xs_auto, ys_auto, xs_fixed, ys_fixed, ys_exact):
    """
    Выводит сравнительную таблицу решений и погрешностей.
    """
    cols = ["x", "y (Авто-шаг)", "y (Пост. шаг)", "y (Эталон)", "Погр. (Авто)", "Погр. (Пост.)"]
    rows = []

    # Так как массивы x для авто и постоянного шага разные,
    # для сравнения возьмем сетку постоянного шага (или можно сделать интерполяцию).
    # Будем выводить значения для точек постоянного шага.

    for i in range(len(xs_fixed)):
        x = xs_fixed[i]
        y_fix = ys_fixed[i]
        y_ex = ys_exact[i]

        # Для авто-шага находим ближайшую точку (или интерполируем)
        # Так как авто-шаг может не попадать ровно в узлы постоянного,
        # найдем индекс ближайшего значения x.
        idx_auto = (np.abs(xs_auto - x)).argmin()
        y_auto = ys_auto[idx_auto]
        x_auto_actual = xs_auto[idx_auto]

        err_auto = abs(y_auto - y_ex)
        err_fix = abs(y_fix - y_ex)

        rows.append([
            f"{x:.3f}",
            f"{y_auto:.5f} (x={x_auto_actual:.3f})",
            f"{y_fix:.5f}",
            f"{y_ex:.5f}",
            f"{err_auto:.2e}",
            f"{err_fix:.2e}"
        ])

    Rich.print_table(
        title="Сравнение численных методов с эталонным решением",
        cols=cols,
        rows=rows
    )


def main():
    """
    Запуск поиска решений дифференциального уравнения и построения графиков
    :return: None
    """
    xs_auto, ys_auto, h_final = solve_auto_step(x0, b, y0, h0, eps)

    N = len(xs_auto) - 1
    """Количество шагов"""

    Rich.debug_log("Решение с автоподбором шага")
    Rich.simple_log(f"Количество шагов N: {N}")
    print_steps_table(xs_auto, ys_auto)

    # Получаем решение с постоянным шагом
    Rich.debug_log("Решение с фиксированным шагом")
    xs_fixed, ys_fixed, h_fixed = solve_with_fixed_step(x0, b, y0, N)

    print_fixed_steps_table(xs_fixed, ys_fixed)

    # Аналитическое решение (через solve_ivp для проверки)
    sol = solve_ivp(lambda x, y: f(x, y), [x0, b], [y0], t_eval=xs_fixed, method='RK45')
    ys_exact = sol.y[0]

    print_comparison_table(xs_auto, ys_auto, xs_fixed, ys_fixed, ys_exact)

    max_err_auto = np.max(np.abs(np.interp(xs_fixed, xs_auto, ys_auto) - ys_exact))
    max_err_fixed = np.max(np.abs(ys_fixed - ys_exact))
    Rich.success_log(f"Максимальная погрешность (Авто): {max_err_auto:.2e}")
    Rich.success_log(f"Максимальная погрешность (Пост.): {max_err_fixed:.2e}")

    return xs_auto, ys_auto, xs_fixed, ys_fixed


xs_auto, ys_auto, xs_fixed, ys_fixed = main()


def show_graphics():
    """Функция для показа графиков на матплотлиб"""
    plt.figure(figsize=(15, 5))

    # График 1: Автоматический шаг
    plt.subplot(1, 3, 1)
    plt.plot(xs_auto, ys_auto, 'bo-', label='Авто-шаг')
    plt.title('Автоматический шаг')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()

    # График 2: Постоянный шаг
    plt.subplot(1, 3, 2)
    plt.plot(xs_fixed, ys_fixed, 'rs--', label='Постоянный шаг')
    plt.title('Постоянный шаг')
    plt.xlabel('x')
    plt.grid(True)
    plt.legend()

    # График 3: Совмещённый
    plt.subplot(1, 3, 3)
    plt.plot(xs_auto, ys_auto, 'bo-', label='Авто-шаг', markersize=4)
    plt.plot(xs_fixed, ys_fixed, 'rs--', label='Постоянный шаг', markersize=4)
    plt.title('Совмещённый график')
    plt.xlabel('x')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_graphics()
