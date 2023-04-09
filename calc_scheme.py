from abc import ABCMeta

import numpy as np


class BaseScheme(metaclass=ABCMeta):
    """
    Базовый класс для описания вычислительной схемы
    """
    def __init__(self, n, length=1):
        self.h = length / (n-1)
        self.n = n
        self.mesh = self.create_mesh()

    def get_system(self):
        """
        Итоговая СЛАУ
        :return: (A, B)
        """

    def create_matrix(self):
        """
        Создает матрицу коэфициэнтов
        """

    def get_boundary_conditions(self):
        """
        Граничные условия
        """

    def get_free_part(self):
        """
        Возвращает столбец правой части f[i]
        """

    def create_mesh(self):
        """
        Создает сетку
        """


class HeatInTheRod(BaseScheme):
    """
    Вычислительная схема для задачи одномерного распределения тепла в стержне
    """
    def __init__(self, *args, k=1, **kwargs):
        super(HeatInTheRod, self).__init__(*args, **kwargs)
        self.k = k

    def create_mesh(self):
        return np.array([i*self.h for i in range(self.n)], dtype=float)

    def get_exact_solution(self):
        return lambda x: x - np.sin(16 * np.pi * x) / 16

    def get_free_part(self):
        return 16 * np.pi * np.pi * np.sin(16 * np.pi * self.mesh)

    def get_boundary_conditions(self):
        return 0, 1

    def get_initial_conditions(self):
        return np.zeros(self.n, dtype=float)

    @property
    def coefficients(self):
        """
        :return: (doun, main, up)
        """
        return self.k / (self.h * self.h), -2 * self.k / (self.h * self.h), self.k / (self.h * self.h)

    def create_matrix(self):
        b, a, c = self.coefficients

        matrix = np.zeros((self.n, self.n))
        rows, cols = np.indices(matrix.shape)
        matrix[np.diag(rows, k=1), np.diag(cols, k=1)] = c
        matrix[np.diag(rows, k=-1), np.diag(cols, k=-1)] = b
        np.fill_diagonal(matrix, a)

        matrix[0] = matrix[self.n - 1] = np.zeros(self.n, dtype=float)

        matrix[0, 0] = matrix[self.n-1, self.n-1] = 1

        return matrix

    def get_system(self):
        A = self.create_matrix()
        B = self.get_free_part()

        begin, end = self.get_boundary_conditions()

        B[0] = begin
        B[self.n - 1] = end

        return A, B




