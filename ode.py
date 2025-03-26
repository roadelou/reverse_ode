import numpy as np
import torch	# Used to convert the data to correct format.
from scipy.integrate import solve_ivp

# The time interval we work on.
T = np.linspace(0, 1, num=1000)

# Our random number generator for coefficients.
GENERATOR = np.random.default_rng()

def ode_matrix(n, coefs):
	# We make a new diagnoal matrix.
	m = np.eye(n, k=-1)
	# We fill the coefficients in the first row.
	m[0] = coefs
	# We return the matrix.
	return m

def ode_function(t, y, matrix):
	return np.matmul(matrix, y)

def solve_ode(n, all_coefs):
	# We split the coefficents and the initial conditions.
	coefs, y0 = all_coefs[:n], all_coefs[n:]
	# We make the matrix with the equation to solve.
	m = ode_matrix(n, coefs)
	# We send the system to scipy.
	solution = solve_ivp(
		fun = ode_function,
		t_span = (0, 1),
		y0 = y0,
		t_eval = T,
		args = (m,)
	)
	#
	# Error handling.
	if not solution.success:
		raise ValueError(solution.message)
	#
	# We return the last row which contain the value of the function, we discard
	# the derivatives.
	return solution.y[-1]

def draw_sample(n, max_coefs, max_eps):
	#
	# We draw the coefficients for the reference equation. We need 2n
	# coefficients since we are also drawing the initial conditions.
	ref_coefs = GENERATOR.uniform(low=-max_coefs, high=max_coefs, size=2*n)
	# We draw the errored coefficients.
	error_eps = GENERATOR.uniform(low=-max_eps, high=max_eps, size=2*n)
	error_coefs = ref_coefs + error_eps
	# We solve both equations.
	ref_val = solve_ode(n, ref_coefs)
	error_val = solve_ode(n, error_coefs)
	# We return all the data for the dataset.
	return ref_coefs, error_coefs, ref_val, error_val
