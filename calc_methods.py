from abc import ABCMeta

import numpy as np


class BaseCalcMethodSystemEquations(metaclass=ABCMeta):
    """
    Базовый класс для вычисления СЛАУ
    """
    def __init__(self, A, B):
        self.dim = B.shape[0]
        self.matrix = A
        self.vector = B

    def pre_run(self):
        """
        Подготовка матрицы и вектора к решению

        :return: None
        """

    def run(self):
        """
        Основной метод расчета

        :return: Столбец решения
        """

    def post_run(self, x):
        """
        Обработка решения перед показом

        :param x: Столбец решения
        """
        pass

    def calculate(self):
        """
        Запускает расчет по методу

        :return: Столбец решения
        """
        self.pre_run()
        x = self.run()
        self.post_run(x)

        return x


class GaussCalcMethod(BaseCalcMethodSystemEquations):
    """
    Метод решения Гаусса.
    """
    def run(self):
        # Прямой ход
        for k in range(self.dim):
            self.vector[k] /= self.matrix[k, k]
            self.matrix[k] /= self.matrix[k, k]
            for i in range(k + 1, self.dim):

                d = self.matrix[i, k]
                self.vector[i] -= self.vector[k] * d

                for j in range(k, self.dim):
                    self.matrix[i, j] -= d * self.matrix[k, j]

        # Обратный ход
        for k in range(self.dim - 1, -1, -1):
            for i in range(k - 1, -1, -1):
                self.vector[i] -= self.matrix[i, k] * self.vector[k]
                self.matrix[i, k] -= self.matrix[i, k]

        return self.vector


class SORCalcMethod(BaseCalcMethodSystemEquations):
    """
    Метод решения SOR.
    """
    def __init__(self, A, B, x0, omega=1.6, precision=10e-6, max_iter=300000):
        super().__init__(A, B)
        self.precision = precision
        self.max_iter = max_iter
        self.x0 = x0
        self.omega = omega
        self.iter = 0
        self.reration = []

    def run(self):
        x = self.x0.copy()
        error = 1
        x_new = np.copy(x)
        while error > self.precision and self.iter < self.max_iter:
            for i in range(self.dim):
                new_values_sum  = np.dot(self.matrix[i,:i], x_new[:i])
                old_values_sum  = np.dot(self.matrix[i, i+1:], x[i+1:])
                x_new[i] = self.omega * (
                        self.vector[i]
                        - new_values_sum
                        - old_values_sum
                ) / self.matrix[i][i] + (1 - self.omega) * x[i]

            self.iter += 1
            x = x_new
            error = np.linalg.norm(np.dot(self.matrix, x) - self.vector)
            self.reration.append(error)

        return x


