from abc import ABCMeta

import numpy as np


class BaseScheme(metaclass=ABCMeta):
    """
    Базовый класс для описания вычислительной схемы
    """
    def __init__(self, n, tay=0, L=1):
        self.h = L / (n-1)
        self.tay = tay
        self.n = n
        self.create_mesh()

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
        Возвращает столбец возмужений
        """

    def create_mesh(self):
        """
        Создает сетку
        """


class HeatInTheRod(BaseScheme):
    """
    Вычислительная схема для задачи одномерного распределения тепла в стержне
    """
    def create_mesh(self):
        self.mesh = np.array([i*self.h for i in range(self.n)], dtype=float)

    def get_free_part(self):
        return np.sin(np.pi * self.mesh)

    def get_boundary_conditions(self):
        return (0, 1)

    def get_initial_conditions(self):
        return np.zeros(self.n, dtype=float)

    @property
    def coefficients(self):
        """
        :return: (doun, main, up)
        """
        return (-1, 2, -1)

    def create_matrix(self):
        b, a, c = self.coefficients

        matrix = np.zeros((self.n, self.n))
        rows, cols = np.indices(matrix.shape)
        matrix[np.diag(rows, k=1), np.diag(cols, k=1)] = c
        matrix[np.diag(rows, k=-1), np.diag(cols, k=-1)] = b
        np.fill_diagonal(matrix, a)

        return matrix

    def get_system(self):
        A = self.create_matrix()
        B = self.get_free_part()

        b, a, c = self.coefficients
        begin, end = self.get_boundary_conditions()

        B[0] -= b * begin
        B[self.n - 1] -= c * end

        return A, B




