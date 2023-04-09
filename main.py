import numpy as np
import matplotlib.pyplot as plt
from time import (
    time,
)

from calc_methods import SORCalcMethod, GaussCalcMethod
from calc_scheme import HeatInTheRod


def print_sol(x, exact_fn, *numerical):
    """
    Рисует решение
    """
    a = np.arange(0, 1.01, 0.01)
    plt.plot(a, exact_fn(a), label='exact')
    for sol, label, marker in numerical:
        plt.plot(x, sol, marker, label=label)

    plt.xlabel(r'$x$', fontsize=14)
    plt.ylabel(r'$u(x)$', fontsize=14)
    plt.grid(True)
    plt.legend(loc='best', fontsize=12)
    plt.savefig(f'figure/plot_{len(x)}.png')
    plt.show()


def print_reration(calc_obj):
    """
    Рисует норму невязки
    :param calc_obj:
    :return:
    """
    plt.semilogy(range(calc_obj.iter), calc_obj.reration)
    plt.xlabel(r'$iter$', fontsize=14)
    plt.ylabel(r'$norm$', fontsize=14)
    plt.grid(True)
    plt.savefig(f'figure/reration_{calc_obj.dim}.png')
    plt.show()


if __name__ == '__main__':

    scheme = HeatInTheRod(500)
    a, b = scheme.get_system()
    x0 = scheme.get_initial_conditions()

    t0 = time()
    sor_calc_abj = SORCalcMethod(a.copy(), b.copy(), x0)
    sor_solution = sor_calc_abj.calculate()
    t1 = time()
    gaus_solution = GaussCalcMethod(a.copy(), b.copy()).calculate()
    t2 = time()

    print(f'время Гаусс: {t2 - t1}')
    print(f'время SOR: {t1 - t0}')

    print_reration(sor_calc_abj)

    """print_sol(
        scheme.mesh,
        scheme.get_exact_solution(),
        (sor_solution, 'SOR', 'r*'),
        (gaus_solution, 'Gauss', 'kx')
    )
    """