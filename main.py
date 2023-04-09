import numpy as np

from calc_methods import SORCalcMethod
from calc_scheme import HeatInTheRod
from tests import test1

if __name__ == '__main__':
    # test1()

    scheme = HeatInTheRod(20)
    a, b = scheme.get_system()
    x0 = scheme.get_initial_conditions()

    x = SORCalcMethod(a.copy(), b.copy(), x0).calculate()
    print(b)
    print('\n \n')
    print(np.dot(a, x))
