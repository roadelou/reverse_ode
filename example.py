from ode import solve_ode, T
import matplotlib.pyplot as plt
import numpy as np
import math

C = np.array([0.5, -1, 0, -1])
Y = solve_ode(2, C)

def analytical(t):
	mult = math.exp(t / 4)
	mult_sin = math.sqrt(15) / 15
	sin_val = math.sin(math.sqrt(15) * t / 4)
	cos_val = math.cos(math.sqrt(15) * t / 4)
	return (mult_sin * sin_val - cos_val) * mult

Ya = np.array([analytical(t) for t in T])

Ce = C + np.array([0, -0.05, 0, 0.05])
Ye = solve_ode(2, Ce)

plt.clf()
plt.plot(T, Y)
plt.title("Solution of f'' - 0.5f' + f = 0, f(0) = -1, f'(0) = 0")
plt.savefig("img/time_serie.png")

plt.clf()
plt.plot(T,  Y, label="f'' - 0.5f' + f = 0, f(0) = -1, f'(0) = 0")
plt.plot(T, Ye, label="f'' - 0.5f' + 0.95f = 0, f(0) = -0.95, f'(0) = 0")
plt.title("Actual solution and close guess")
plt.legend()
plt.savefig("img/close_guess.png")
