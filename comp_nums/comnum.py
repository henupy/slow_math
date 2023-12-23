"""
Basic operations and such for complex numbers
"""

from __future__ import annotations

import math
import matplotlib.pyplot as plt


class ComplexNumber:
    def __init__(self, real: int | float, imag: int | float) \
            -> None:
        """
        Object for representing complex numbers
        :param real: The real part of the complex number
        :param imag: The imaginary part of the complex number
        """
        self.real = real
        self.imag = imag

    @staticmethod
    def _sign(num: int | float) -> int:
        """
        Returns the sign of num as 1 or -1
        :param num:
        :return:
        """
        if num >= 0:
            return 1
        return -1

    @staticmethod
    def conjugate(com_num: ComplexNumber) -> ComplexNumber:
        return ComplexNumber(com_num.real, -com_num.imag)

    def __add__(self, other: ComplexNumber) -> ComplexNumber:
        sum_real = self.real + other.real
        sum_imag = self.imag + other.imag
        return ComplexNumber(sum_real, sum_imag)

    def __sub__(self, other: ComplexNumber) -> ComplexNumber:
        sum_real = self.real - other.real
        sum_imag = self.imag - other.imag
        return ComplexNumber(sum_real, sum_imag)

    def __mul__(self, other: ComplexNumber) -> ComplexNumber:
        add_real = self.real * other.real - self.imag * other.imag
        add_imag = self.real * other.imag + self.imag * other.real
        return ComplexNumber(add_real, add_imag)

    def __truediv__(self, other: ComplexNumber) -> ComplexNumber:
        num = self * self.conjugate(other)
        denum = other * self.conjugate(other)
        denum = float(denum.real)
        return ComplexNumber(num.real / denum, num.imag / denum)

    def __abs__(self) -> float:
        return math.sqrt(self.real * self.real + self.imag * self.imag)

    def __eq__(self, other: ComplexNumber) -> bool:
        return self.real == other.real and self.imag == other.imag

    def __lt__(self, other: ComplexNumber) -> bool:
        return abs(self) < abs(other)

    def __le__(self, other: ComplexNumber) -> bool:
        return abs(self) <= abs(other)

    def __gt__(self, other: ComplexNumber) -> bool:
        return abs(self) > abs(other)

    def __ge__(self, other: ComplexNumber) -> bool:
        return abs(self) >= abs(other)

    def polar(self) -> str:
        """
        Returns the representation in polar coordinates
        :return:
        """
        r = math.sqrt(self.real * self.real + self.imag * self.imag)
        a = math.degrees(math.atan2(self.imag, self.real))
        return f"{r:.2f}<{a:.2f}"

    def __repr__(self) -> str:
        if self._sign(self.imag) == 1:
            return f"{self.real} + {self.imag}i"
        return f"{self.real} - {self.imag * -1}i"

    def __str__(self) -> str:
        if self._sign(self.imag) == 1:
            return f"{self.real} + {self.imag}i"
        return f"{self.real} - {self.imag * -1}i"


def main():
    comp_1 = ComplexNumber(-1, 5)
    comp_2 = ComplexNumber(3, -10)
    comp_3 = comp_1 * comp_2
    print(f"1: {comp_1.polar()}, 2: {comp_2.polar()}, 3: {comp_3.polar()}")
    plt.scatter(comp_1.real, comp_1.imag, label=repr(comp_1))
    plt.scatter(comp_2.real, comp_2.imag, label=repr(comp_2))
    plt.scatter(comp_3.real, comp_3.imag, label=repr(comp_3))
    plt.xlabel("Real")
    plt.ylabel("Imag")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
