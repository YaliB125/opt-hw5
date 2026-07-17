import numpy as np
import matplotlib.pyplot as plt

def coordinate_descent(H, g, a, b, x0, max_iter, epsilon):
    n = x0.shape[0]

    def f(x):
        return 1/2 * x.T @ H @ x - x.T @ g
    
    def grad_f(x):
        return H @ x - g
    
    f_values = []
    iterations = []
    projected_gradient_norms = []

    x = x0.copy()

    for k in range(max_iter):
        iterations.append(k)
        f_values.append(f(x))

        y = x - grad_f(x)
        projected_y = np.clip(y, a, b)
        projected_gradient_norms.append(np.linalg.norm(x - projected_y))

        new_x = np.copy(x)

        for i in range(n):
            tmp = g[i] - np.dot(H[i], new_x) + H[i, i] * new_x[i]
            val = tmp / H[i, i]
            new_x[i] = np.clip(val, a[i], b[i])
        
        if np.linalg.norm(new_x - x) <= epsilon * (np.linalg.norm(x) + 1e-12):
            """ 
            not doing "if np.linalg.norm(new_x - x) / np.linalg.norm(x) < epsilon" 
            to aviod possible divition in zero
            """
            break

        x = new_x
    
    return x, projected_gradient_norms, f_values, iterations

def coordinate_descent_graphs(H, g, a, b, x0, max_iter, atol):
    sol, projected_gradient_norms, f_values, iterations = coordinate_descent(H, g, a, b, x0, max_iter, atol)

    plt.semilogy(iterations, projected_gradient_norms)
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(
        r"$\|x^{(k)}-\Pi_{a\leq x\leq b}\left(x^{(k)}-\nabla f\right)\|_2$"
    )
    plt.yscale("log")
    plt.title(f"projected gradient step norm")
    plt.grid(True)
    plt.show()

    plt.plot(iterations, f_values)
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"$f(x^{(k)})$")
    plt.title(f"Value of f - CD Method")
    plt.grid(True)
    plt.show()

    return sol

def main():
    H = np.array([
        [5, -1, -1, -1, -1],
        [-1, 5, -1, -1, -1],
        [-1, -1, 5, -1, -1],
        [-1, -1, -1, 5, -1],
        [-1, -1, -1, -1, 5]
    ])

    g = np.array([
        18,
        6,
        -12,
        -6,
        18
    ])

    a = np.zeros(5)

    b = 5 * np.ones(5)

    sol = coordinate_descent_graphs(H, g, a, b, np.zeros(5), 5000, 10e-3)

    print("sol:", sol)

main()

# sol: [5.         3.63885584 0.65112719 1.65799661 5.        ]
# [5, 3.63885584, 0.65112719, 1.65799661, 5]