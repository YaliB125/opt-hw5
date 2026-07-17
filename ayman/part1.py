import numpy as np
import matplotlib.pyplot as plt

def armijo_linesearch(f ,grad_f, constraints_funcs, x, d, alpha0, beta, max_iter):
    c = 1e-4
    alpha = alpha0

    fx = f(x)
    gx = grad_f(x)

    def is_in_feasible_region(x):
        for constraint_func in constraints_funcs:
            if constraint_func(x) > 0:
                return False
        
        return True

    for _ in range(max_iter):
        x_new = x + alpha * d

        if is_in_feasible_region(x_new) and f(x_new) <= fx + c* alpha * np.dot(gx, d):
            return alpha
        
        alpha = beta * alpha

    return alpha

def gradient_descent(f, grad_f, constraints_funcs, active_constraints_idxs, x0, max_iter, epsilon):
    gradient_norm_first_sol = np.linalg.norm(grad_f(x0))

    solutions_norm_of_the_differences_from_zero = []
    f_values = []
    iterations = []

    x = x0

    def norm_of_the_differences_from_zero(x):
        values = np.array([constraints_funcs[i](x) for i in active_constraints_idxs])

        return np.linalg.norm(values)


    for k in range(max_iter):
        iterations.append(k)
        f_values.append(f(x))

        g = grad_f(x)
        d = -g

        solutions_norm_of_the_differences_from_zero.append(norm_of_the_differences_from_zero(x))

        gradient_norm_curr_sol = np.linalg.norm(g)         
        if gradient_norm_curr_sol <= epsilon * (gradient_norm_first_sol + 1e-12):
            """ 
            not doing "if gradinet_norm_curr_sol / gradinet_norm_first_sol < epsilon" 
            to aviod possible divition in zero
            """
            return x, solutions_norm_of_the_differences_from_zero, f_values, iterations
        
        armijo_alpha = armijo_linesearch(f, grad_f, constraints_funcs, x, d, alpha0=0.5, beta=0.5, max_iter=20)
        x = x + armijo_alpha * d
    
    return x, solutions_norm_of_the_differences_from_zero, f_values, iterations

def gradient_descent_graphs(f, grad_f, constraints_funcs, active_constraints_idxs, x0, max_iter, atol, parameter):
    sol, solutions_norm_of_the_differences_from_zero, f_values, iterations = gradient_descent(f, grad_f, constraints_funcs, active_constraints_idxs, x0, max_iter, atol)

    plt.figure(1)
    plt.semilogy(
        iterations,
        solutions_norm_of_the_differences_from_zero,
        label=f"c={parameter}"
    )

    plt.figure(2)
    plt.plot(
        iterations,
        f_values,
        label=f"c={parameter}"
    )

    return sol

def main():
    def f(x):
        x1, x2 = x

        return x1 + 0.5 * x2
    
    def grad_f(x):
        _, _ = x

        return np.array([1, 0.5])
    
    def constraint1(x):
        x1, x2 = x

        return x1 + x2 - 1
    
    def grad_constaint1(x):
        _, _ = x

        return np.array([1, 1])

    def constraint2(x):
        x1, x2 = x

        return x1 - x2 - 1

    def grad_constaint2(x):
        _, _ = x

        return np.array([1, -1])

    def constraint3(x):
        x1, x2 = x

        return (-1 * x1) + x2 - 1

    def grad_constaint3(x):
        _, _ = x

        return np.array([-1, 1])

    def constraint4(x):
        x1, x2 = x

        return (-1 * x1) - x2 - 1

    def grad_constaint4(x):
        _, _ = x

        return np.array([-1, -1])

    def constraint5(x):
        x1, _ = x

        return -1 * x1

    def grad_constaint5(x):
        _, _ = x

        return np.array([-1, 0])

    def constraint6(x):
        _, x2 = x

        return -1 * x2

    def grad_constaint6(x):
        _, _ = x

        return np.array([0, -1])
    
    def penalized_f(c):
        def penalized_f_parameter(x):
            log_barrier_constraints = [
                np.log(-constraint1(x)),
                np.log(-constraint2(x)),
                np.log(-constraint3(x)),
                np.log(-constraint4(x)),
                np.log(-constraint5(x)),
                np.log(-constraint6(x)),
            ]

            return f(x) - c * sum(log_barrier_constraints)
        
        return penalized_f_parameter
    
    def grad_penalized_f(c):
        def grad_penalized_f_parameter(x):
            grad_log_barrier_constraints = [
                grad_constaint1(x) / constraint1(x),
                grad_constaint2(x) / constraint2(x),
                grad_constaint3(x) / constraint3(x),
                grad_constaint4(x) / constraint4(x),
                grad_constaint5(x) / constraint5(x),
                grad_constaint6(x) / constraint6(x),
            ]

            return grad_f(x) - c * sum(grad_log_barrier_constraints) 
        
        return grad_penalized_f_parameter


    constraints_funcs = [
        constraint1,
        constraint2,
        constraint3,
        constraint4,
        constraint5,
        constraint6
    ]
    active_constraints_idxs = [4, 5]
    parameters = [10, 1, 0.1, 0.01, 0.001]

    max_iter = 1000
    x0 = np.array([0.2, 0.2])
    atol = 10e-6

    for parameter in parameters:
        sol = gradient_descent_graphs(penalized_f(parameter), grad_penalized_f(parameter), constraints_funcs, active_constraints_idxs, x0, max_iter, atol, parameter)

        print(f"sol for parameter {parameter} is: {np.round(sol, 2)}")
    
    plt.figure(1)
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"$\|(c^{ieq}_5(x^{(k)}))^2+(c^{ieq}_6(x^{(k)}))^2\|$")
    plt.title("Norm of the active constraints")
    plt.grid(True)
    plt.legend()

    plt.figure(2)
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"$penalized\_func(x^{(k)})$")
    plt.title("Penalized objective")
    plt.grid(True)
    plt.legend()

    plt.show()

main()

"""
sol for parameter 10 is: [0.35 0.35]
sol for parameter 1 is: [0.31 0.35]
sol for parameter 0.1 is: [0.1  0.17]
sol for parameter 0.01 is: [0.01 0.02]
sol for parameter 0.001 is: [0. 0.]
"""
