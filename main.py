import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


def cost_function(t, x, u):
    
    L = np.sin(t) * x**2 + np.exp(-u)
    return L


def system_equations(t, x, u):
    
    f = np.cos(t) * x**2 + u
    return f



t1 = 0.4
t2 = 0.8
x1 = 0.0  
x2 = 0.8  
x_limits = (0.4, 1.0)  
u_limits = (-2.0, 2.0)  




def objective_function(u):
    sol = solve_ivp(system_equations, [t1, t2], [
                    x1], args=(u,), dense_output=True)

    t = np.linspace(t1, t2, 100)
    L_values = cost_function(t, sol.sol(t), u)
    J = np.trapz(L_values, t)

    return J

constraint = [{'type': 'eq', 'fun': lambda u: solve_ivp(system_equations, [t1, t2], [x1], args=(u,), dense_output=True).sol(t2)[0] - x2},{'type': 'ineq', 'fun': lambda x: x - x_limits[0]},{'type': 'ineq', 'fun': lambda x: x_limits[1] - x},{'type': 'ineq', 'fun': lambda u: u - u_limits[0]},{'type': 'ineq', 'fun': lambda u: u_limits[1] - u}]

result = minimize(objective_function, np.zeros(10), bounds=[u_limits]*10, constraints=constraint)


u_opt = result.x


print("Optimal control:", u_opt)
print("Optimal cost:", result.fun)
