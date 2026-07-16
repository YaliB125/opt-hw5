import numpy as np
import matplotlib.pyplot as plt

H = np.array([
    [5, -1, -1, -1, -1],
    [-1, 5, -1, -1, -1],
    [-1, -1, 5, -1, -1],
    [-1, -1, -1, 5, -1],
    [-1, -1, -1, -1, 5]
], dtype=float)

g = np.array([18, 6, -12, -6, 18], dtype=float)
a = 0.0 
b = 5.0 
n = len(g)

x = np.zeros(n)       
max_iter = 100         
tolerance = 1e-3       

obj_values = []
proj_grad_norms = []

def calc_objective(x):
    return 0.5 * np.dot(x.T, np.dot(H, x)) - np.dot(x.T, g)

def calc_proj_grad_norm(x):
    grad = np.dot(H, x) - g
    proj_step = np.clip(x - grad, a, b)
    return np.linalg.norm(x - proj_step)

for k in range(max_iter):
    x_old = x.copy()
    
    for i in range(n):
        sum_hx = np.dot(H[i, :], x) - H[i, i] * x[i]
        
        x_unc = (g[i] - sum_hx) / H[i, i]
        
        x[i] = np.clip(x_unc, a, b)
        
    obj_values.append(calc_objective(x))
    proj_grad_norms.append(calc_proj_grad_norm(x))
    
    rel_change = np.linalg.norm(x - x_old) / (np.linalg.norm(x_old) + 1e-10)
    
    if rel_change < tolerance:
        print(f"Algorithm converged successfully after {k+1} iterations!")
        break

print("Final Solution (x*):", x.round(4))
print("Final Objective Value:", round(obj_values[-1], 4))

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(obj_values, marker='o', linestyle='-', color='b', linewidth=2)
plt.title('Objective Function vs. Iterations')
plt.xlabel('Iteration (k)')
plt.ylabel('f(x)')
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
plt.semilogy(proj_grad_norms, marker='s', linestyle='-', color='r', linewidth=2)
plt.title('Projected Gradient Step Norm vs. Iterations')
plt.xlabel('Iteration (k)')
plt.ylabel('||x - Proj(x - grad)||')
plt.grid(True, which="both", linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()