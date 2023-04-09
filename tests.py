import numpy as np

from calc_methods import SORCalcMethod


def test1():
    A = np.random.sample((3, 3))
    B = np.random.sample(3)
    A1 = np.array([
        [1, 1, 1],
        [1, 3, 0],
        [1, 0, 1]
    ], dtype=float)
    B1 = np.array([4, 3, 1], dtype=float)

    # x = GaussCalcMethod(A.copy(), B.copy()).calculate()

    a = np.array([
        [31., -13., 0., 0., 0., -10., 0., 0., 0.],
        [-13., 35., 9., 0., -11, 0., 0., 0., 0.],
        [0., -9., 31, -10., 0., 0., 0., 0., 0.],
        [0., 0., -10., 79., -30., 0., 0., 0., -9.],
        [0., 0., 0., -30., 57., -7., 0., -5., 0.],
        [0., 0., 0., 0., -7., 47., -30., 0., 0.],
        [0., 0., 0., 0., 0., -30., 41., 0., 0.],
        [0., 0., 0., 0., -5., 0., 0., 27, -2.],
        [0., 0., 0., -9., 0., 0., 0., -2., 29.],
    ])
    b = np.array([-15., 27., -23., 0., -20., 12., -7., 7., 10])

    x0 = np.zeros(9, dtype=float)
    x = SORCalcMethod(a.copy(), b.copy(), x0).calculate()
    print(b)
    print('\n \n')
    print(np.dot(a, x))
