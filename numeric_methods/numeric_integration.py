"""
Some ways to integrate numerically
"""

import sys
import math
import random


class NumInt:
    def __init__(self, f: callable, a: int | float,
                 b: int | float) -> None:
        """
        :param f: The function to be integrated, must be an actual
        python-function
        :param a: Lower integration limit
        :param b: Upper integration limit
        """
        self.f = f
        self.a = a
        self.b = b

    def mc(self, num_points: int, y_max: float) -> float:
        """
        :param num_points: Number of random points to be sampled
        :param y_max: The maximum value of y that defines the box around
        the function
        :return:
        """
        count, in_area = 0, 0
        while count < num_points:
            x_coord = random.uniform(self.a, self.b)
            y_coord = random.uniform(0, y_max)
            if y_coord < self.f(x_coord):
                in_area += 1
            count += 1
        total_area = (self.b - self.a) * y_max
        area = in_area / count * total_area
        return area

    def rectangles(self, num_rects: int) -> float:
        """
        Calculates the integral by dividing the area under the curve
        to small rectangles, and summing the area of the rectangles
        :param num_rects: Number of rectangles
        :return:
        """
        h = (self.b - self.a) / num_rects
        total_area = 0
        for i in range(num_rects):
            xn = self.a + (h / 2) + i * h
            area = self.f(xn) * h
            total_area += area

        return total_area


def logfun(x: int | float) -> float:
    if x == 0:
        x = sys.float_info.epsilon
    return math.log(x) / x


def cosfun(x: int | float) -> float:
    return x * math.cos(x)


def expfun(x: int | float) -> float:
    return x ** 3 * math.exp(x ** 2)


def main():
    a, b = 0, 2
    numint = NumInt(expfun, a, b)
    points_mc = int(1e5)
    rects = 1000
    y_max = expfun(b)
    area_mc = numint.mc(points_mc, y_max)
    area_rect = numint.rectangles(rects)
    print(f'Monte Carlo: {area_mc:.4f}')
    print(f'Rectangle: {area_rect:.4f}')


if __name__ == '__main__':
    main()
