import numpy as np
import matplotlib.pyplot as plt

A = np.array([[1.0, 1.0], 
              [1.0, -1.0], 
              [-1.0, 1.0], 
              [-1.0, -1.0]])
b = np.array([1.0, 1.0, 1.0, 1.0])
c = np.array([1.0, 0.5])

def f_mu(x, mu):
    if np.any(b - np.dot(A, x) <= 0) or np.any(x <= 0):
        return np.inf

    barrier_A = np.sum(np.log(b - np.dot(A, x)))
    barrier_x = np.log(x[0]) + np.log(x[1])
    
    return np.dot(c, x) - mu * barrier_A - mu * barrier_x


def grad_f_mu(x, mu):
    g = np.copy(c)
    
    for i in range(4):
        g += mu * A[i, :] / (b[i] - np.dot(A[i, :], x))
    g[0] -= mu / x[0]
    g[1] -= mu / x[1]
    
    return g


def armijo_linesearch(f, x, d, gk, mu, beta=0.5, max_iter=100):
    c_val = 1e-4
    alpha = 1.0
    for _ in range(max_iter):
        if f(x + alpha * d, mu) <= f(x, mu) + c_val * alpha * np.dot(gk, d):
            return alpha
        alpha = beta * alpha
    return alpha

mu_vec = [10.0, 1.0, 0.1, 0.01, 0.001]
x = np.array([0.2, 0.2]) 

all_f_vals = []
all_active_violations = []

for mu in mu_vec:
    for iter in range(100):
        gk = grad_f_mu(x, mu)
        if np.linalg.norm(gk) < 1e-4:
            break
        d = -gk
        alpha = armijo_linesearch(f_mu, x, d, gk, mu, beta=0.5)
        x = x + alpha * d
        
        all_f_vals.append(f_mu(x, mu))
        all_active_violations.append(np.sqrt(x[0]**2 + x[1]**2))
        
    print(f"Result for mu = {mu}: x = {x}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(all_f_vals)
plt.title("Penalized Function Value")
plt.xlabel("Iterations")
plt.ylabel("Function Value")
plt.subplot(1, 2, 2)
plt.semilogy(all_active_violations)
plt.title("Norm of Active Constraints (Semi-log)")
plt.xlabel("Iterations")
plt.ylabel("Norm")
plt.show()