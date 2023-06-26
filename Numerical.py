import math
import numpy as np
import sympy as sp


def bisection(function, start, end, error):
    """
    Finds the root of a function using the bisection method.

    Args:
        function: The function to find the root of.
        start: The start point of the interval.
        end: The end point of the interval.
        error: The desired error tolerance.

    Raises:
        ValueError: If the function has the same sign at both the start and end points.

    Returns:
        None
    """
    A = start
    B = end
    start_sign = function(A) > 0
    end_sign = function(B) > 0
    i = 0
    while True:
        i += 1
        mid = (A + B) / 2
        mid_sign = function(mid) > 0
        current_error = abs(A - B)

        if mid_sign == start_sign:
            A = mid
        elif mid_sign == end_sign:
            B = mid
        else:
            raise ValueError("Function has the same sign at both the start and end points.")

        if function(mid) == 0:
            print(f"Root found in iteration #{i}. x = {mid} with Error = {current_error}")
            break

        print(f"Iteration #{i}. x = {mid} with Error = {current_error}")

        if current_error <= error:
            break



def newton_method(function, derivative, initial_guess, error, max_iterations=100):
    """
    Finds the root of a function using Newton's method.

    Args:
        function: The function to find the root of.
        derivative: The derivative of the function.
        initial_guess: The initial guess for the root.
        error: The desired error tolerance.
        max_iterations: The maximum number of iterations.

    Raises:
        ValueError: If the derivative is zero at the initial guess.

    Returns:
        None
    """
    x = initial_guess
    i = 0
    while i < max_iterations:
        i += 1
        f_x = function(x)
        f_prime_x = derivative(x)

        if f_prime_x == 0:
            raise ValueError("Derivative is zero at the initial guess.")

        x_next = x - f_x / f_prime_x
        current_error = abs(x_next - x)

        if math.isclose(x_next, 0, abs_tol=error):
            print(f"Root found in iteration #{i}. x = {x_next} with Error = {current_error}")
            break

        print(f"Iteration #{i}. x = {x_next} with Error = {current_error}")

        x = x_next

    if i == max_iterations:
        print("Maximum iterations reached.")


def secant_method(function, x0, x1, error, max_iterations=100):
    """
    Finds the root of a function using the secant method.

    Args:
        function: The function to find the root of.
        x0: The first initial guess.
        x1: The second initial guess.
        error: The desired error tolerance.
        max_iterations: The maximum number of iterations.

    Returns:
        None
    """
    i = 0
    while i < max_iterations:
        i += 1
        f_x0 = function(x0)
        f_x1 = function(x1)

        x_next = x1 - (f_x1 * (x1 - x0)) / (f_x1 - f_x0)
        current_error = abs(x_next - x1)

        if math.isclose(x1, x_next, abs_tol=error):
            print(f"Root found in iteration #{i}. x = {x_next} with Error = {current_error}")
            break

        print(f"Iteration #{i}. x = {x_next} with Error = {current_error}")

        x0 = x1
        x1 = x_next

    if i == max_iterations:
        print("Maximum iterations reached.")



# Example usage
def f(x):
    return x**2

def f_prime(x):
    return 2 * x

#secant_method(f, 1, 2, 0.0001)
#newton_method(f, f_prime, 90, 0.001)
#bisection(f, 0, 180, 0.001)




def my_function():
    """
    Defines a symbolic function using sympy.

    Returns:
        The symbolic expression of the function.
    """
    x = sp.symbols('x')
    return sp.exp(x**2)

def differentiate(func, order: int):
    """
    Differentiates a symbolic expression multiple times.

    Args:
        func: The symbolic expression to differentiate.
        order: The number of times to differentiate the expression.

    Returns:
        The differentiated symbolic expression.
    """
    x = sp.symbols('x')  # Define x as a symbolic variable
    derivative = func
    if order <= 0:
        return derivative
    derivative = derivative.diff(x)  # Differentiate the symbolic expression
    return differentiate(derivative, order - 1)


def callabledifferentiate(func, order: int):
    """
    Converts a symbolic expression to a callable function and differentiates it multiple times.

    Args:
        func: The symbolic expression to differentiate.
        order: The number of times to differentiate the expression.

    Returns:
        The differentiated callable function.
    """
    x = sp.symbols('x')
    derivative = func
    if order <= 0:
        return derivative

    for _ in range(order):
        derivative = derivative.diff(x)

    derivative_func = sp.lambdify(x, derivative)  # Convert the symbolic expression to a callable function

    def derivative_wrapper(x_value):
        return derivative_func(x_value)

    return derivative_wrapper


new_function = differentiate(my_function(), 4)  # Call my_function() to get the symbolic expression
print(f"new f(x) = {new_function}")

derivative_func = callabledifferentiate(my_function(), 4)
result = derivative_func(1)

print(f"f(1) = {result}")


