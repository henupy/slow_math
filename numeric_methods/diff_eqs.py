"""
Some not very well written methods to solve simple (ordinary)
differential equations numerically
"""

import numpy as np
import matplotlib.pyplot as plt

from typing import Callable
from scipy.integrate import odeint

# For typing
num = int | float | list | np.ndarray


def eulerfw(diff_eq: Callable, y0: num, trange: np.ndarray, *args) \
        -> np.ndarray:
    """
    Solves a system of first-order ordinary differential equations
    with the provided parameters and initial conditions numerically
    using the (forward) Euler method.
    :param diff_eq: Function for y'
    :param y0: Initial values for the equations
    :param trange: Time points where the equation is solved
    :param args: Other arguments for the differential equation
    :return: A m x n size matrix, where m is the amount of equations
        and n is the amount of timesteps. Contains values for each equation
        at each timestep.
    """
    m = trange.size
    if not isinstance(y0, np.ndarray):
        y0 = np.array(y0)
    n = y0.size
    dt = trange[1] - trange[0]
    sol = np.zeros((m, n))
    sol[0, :] = y0
    for i, t in enumerate(trange[1:], start=1):
        y_vals = dt * diff_eq(sol[i - 1, :], trange[i - 1], *args)
        sol[i, :] = sol[i - 1, :] + y_vals
    return sol


def eulerbw(diff_eq: Callable, y0: num, trange: np.ndarray, *args) \
        -> np.ndarray:
    """
    Solves a first-order ordinary differential equation with the
    provided parameters and initial conditions numerically using the
    backward Euler method.
    :param diff_eq: Function for y'
    :param y0: Initial values for the equations
    :param trange: Time points where the equation is solved
    :param args: Other arguments for the differential equation
    :return: A m x n size matrix, where m is the amount of equations
        and n is the amount of timesteps. Contains values for each equation
        at each timestep.
    """
    if not isinstance(y0, np.ndarray):
        y0 = np.array(y0)
    m, n = trange.size, y0.size
    sol = np.zeros((m, n))
    sol[0, :] = y0
    dt = trange[1] - trange[0]
    for i, t in enumerate(trange[1:], start=1):
        y_vals = dt * diff_eq(sol[i - 1, :], trange[i - 1], *args)
        yt1 = sol[i - 1, :] + y_vals
        count = 0
        yt2 = np.zeros(n)
        while count < 20:
            yt2 = sol[i - 1, :] + dt * diff_eq(yt1, trange[i], *args)
            yt1 = yt2
            count += 1
        sol[i, :] = yt2
    return sol


def rk4(diff_eq: Callable, y0: num, trange: np.ndarray, *args) -> np.ndarray:
    """
    Solves a system of first-order differential equations using the
    classic Runge-Kutta method
    :param diff_eq: Function which returns the righthandside of the
        equations making up the system of equations
    :param y0: Initial values for y and y' (i.e. the terms in the system
        of equations)
    :param trange: Time points for which the equation is solved
    :param args: Any additional paramaters for the differential equation
    :return: A m x n size matrix, where m is the amount of equations
        and n is the amount of timesteps. Contains values for each equation
        at each timestep.
    """
    m = trange.size
    dt = trange[1] - trange[0]
    if not isinstance(y0, np.ndarray):
        y0 = np.array(y0)
    n = y0.size
    sol = np.zeros((m, n))
    sol[0, :] = y0
    for i, t in enumerate(trange[1:], start=1):
        y = sol[i - 1, :]
        k1 = diff_eq(y, t, *args)
        k2 = diff_eq(y + dt * k1 / 2, t + dt / 2, *args)
        k3 = diff_eq(y + dt * k2 / 2, t + dt / 2, *args)
        k4 = diff_eq(y + dt * k3, t + dt, *args)
        y += 1 / 6 * dt * (k1 + 2 * k2 + 2 * k3 + k4)
        sol[i, :] = y
    return sol


def some_eq(y: int | float | np.ndarray, t: int | float) \
        -> float:
    """
    An ordinary differential equation of order one
    :param y:
    :param t:
    :return: Returns the righthandside of the equation
    y' == -t / (1 + y ** 2)
    """
    return -t / (1 + y * y)


def sin_eq(y: int | float | np.ndarray, t: int | float) -> float:
    """
    An ordinary first order differential equation
    :param y:
    :param t:
    :return:
    """
    return np.sin(t) * np.sin(t) * y


def pend(y0, _, b, c) -> np.ndarray:
    """
    Equation for a simple pendulum
    :param y0:
    :param _:
    :param b:
    :param c:
    :return:
    """
    theta, omega = y0
    dydt = [omega, -b * omega - c * np.sin(theta)]
    return np.array(dydt)


def ait(y, _) -> np.ndarray:
    """
    van der Pol equation from matlab examples
    :param y:
    :param _:
    :return:
    """
    y1, y2 = y
    dydt = [y2, (1 - y1 * y1) * y2 - y1]
    return np.array(dydt)


def yeet():
    # Pendulum
    b, c = 0.25, 5.
    y0 = [np.pi - 0.1, 0.0]
    t = np.linspace(0, 60, 601)
    sol = odeint(pend, y0, t, args=(b, c))
    plt.plot(t, sol[:, 0], 'b', label='theta(t)')
    plt.plot(t, sol[:, 1], 'r', label='omega(t)')
    sol_e = rk4(pend, y0, t, b, c)
    plt.plot(t, sol_e[:, 0], 'g', label='theta(t) rk4')
    plt.plot(t, sol_e[:, 1], 'c', label='omega(t) rk4')

    # # Example from matlab
    # tspan = np.linspace(0, 20, 201)
    # initvals = [2, 0]
    # sol = odeint(ait, initvals, tspan)
    # plt.plot(tspan, sol[:, 0], 'bo-', label='y1', fillstyle='none')
    # plt.plot(tspan, sol[:, 1], 'ro-', label='y2', fillstyle='none')
    # sol_rk = rk4(ait, initvals, tspan)
    # plt.plot(tspan, sol_rk[:, 0], 'go-', label='y1 (rk4)', fillstyle='none')
    # plt.plot(tspan, sol_rk[:, 1], 'co-', label='y2 (rk4)', fillstyle='none')

    plt.legend()
    plt.grid()
    plt.show()


def main():
    t0, max_t = 0, 5
    y0, dt = 1, 0.01
    eq = sin_eq
    y0bw = [1]
    t_vals = np.arange(t0, max_t + dt, dt)
    y_rk = rk4(eq, y0, t_vals)
    y_efw = eulerfw(eq, y0bw, t_vals)
    y_ebw = eulerbw(eq, y0bw, t_vals)
    plt.plot(t_vals, y_rk, 'r', label='y (RK4)')
    plt.plot(t_vals, y_ebw, 'g', label='y (eulerbw)')
    plt.plot(t_vals, y_efw, 'c', label='y (eulerfw)')
    plt.legend()
    plt.grid()
    plt.show()

    # yeet()


if __name__ == '__main__':
    main()
