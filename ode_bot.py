import pickle
import torch
import random
from sklearn.metrics import mean_squared_error
from tqdm import tqdm	# Progress bar
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from common import N, M
from model import MODEL as IMPROV_NN
from oneshot import MODEL as ONESHOT_NN
from ode import solve_ode

with open(f"dataset.{N}.pickle", "rb") as src:
	DATASET = pickle.load(src)

DATASET = random.sample(DATASET, 2000)

# Loading the models.
IMPROV_NN.load_state_dict(torch.load(f"improv.{N}.pth", weights_only=True))
ONESHOT_NN.load_state_dict(torch.load(f"oneshot.{N}.pth", weights_only=True))
IMPROV_NN.eval()
ONESHOT_NN.eval()

with open(f"model.linear.{N}.pickle", "rb") as src:
	ONESHOT_LIN = pickle.load(src)

with open(f"improve.linear.{N}.pickle", "rb") as src:
	IMPROV_LIN = pickle.load(src)

def guess_zero(values):
	return np.zeros(2*N)

def guess_lin(values):
	return ONESHOT_LIN.predict(np.reshape(values, (1, -1)))[0]

def guess_opt(values):
	solution = minimize(
		fun = lambda x: mean_squared_error(solve_ode(N, x), values),
		x0 = np.zeros(2*N),
		# We let the optimizer do its full job for the initial guess.
	)
	if solution.success:
		return solution.x
	else:
		return np.zeros(2*N)

def guess_nn(values):
	with torch.no_grad():
		return ONESHOT_NN(torch.from_numpy(values).to(torch.float)).numpy()

def improv_zero(values, guess_coefs, guess_values):
	return guess_coefs

def improv_lin(values, guess_coefs, guess_values):
	return guess_coefs + IMPROV_LIN.predict(
		np.reshape(values - guess_values, (1, -1))
	)[0]

def improv_nn(values, guess_coefs, guess_values):
	with torch.no_grad():
		return IMPROV_NN(
			torch.from_numpy(guess_coefs).to(torch.float),
			torch.from_numpy(np.stack((
				values,
				guess_values
			))).to(torch.float)
		).numpy()

def improv_opt(values, guess_coefs, guess_values):
	solution = minimize(
		fun = lambda x: mean_squared_error(solve_ode(N, x), values),
		x0 = guess_coefs,
		options = {
			"maxiter": 1
		}
	)
	if solution.success:
		return solution.x
	else:
		return np.zeros(2*N)

# For the whole dataset we try:
# - Base strategy of just guessing zeros.
# - Just oneshot.
# - Oneshot then refine.
# - Zero then refine.
COMBINATIONS = {
	"zero/zero": (guess_zero, improv_zero),
	"zero/lin" : (guess_zero, improv_lin ),
	"zero/nn"  : (guess_zero, improv_nn  ),
	"lin/zero" : (guess_lin , improv_zero),
	"lin/lin"  : (guess_lin , improv_lin ),
	"lin/nn"   : (guess_lin , improv_nn  ),
	"nn/zero"  : (guess_nn  , improv_zero),
	"nn/lin"   : (guess_nn  , improv_lin ),
	"nn/nn"    : (guess_nn  , improv_nn  ),
	"opt/zero" : (guess_opt , improv_zero),
	"opt/lin"  : (guess_opt , improv_lin ),
	"opt/nn"   : (guess_opt , improv_nn  ),
	"opt/opt"  : (guess_opt , improv_opt ),
	"zero/opt" : (guess_zero, improv_opt ),
	"lin/opt"  : (guess_lin , improv_opt ),
	"nn/opt"   : (guess_nn  , improv_opt ),
}

SCORES = dict()
for combination, (guess, improv) in COMBINATIONS.items():
	print(f"Testing the combination: {combination}")
	dataset_score_list = list()
	for ref_coefs, _, ref_val, _ in tqdm(DATASET):
		guess_coefs = guess(ref_val)
		score_list = [mean_squared_error(ref_coefs, guess_coefs)]
		for _ in range(M):
			#
			# We compute the ode.
			guess_val = solve_ode(N, guess_coefs)
			#
			# Improving the guess for the coefficients.
			guess_coefs = improv(ref_val, guess_coefs, guess_val)
			#
			# Adding the score to the list.
			score_list.append(mean_squared_error(ref_coefs, guess_coefs))
		dataset_score_list.append((ref_coefs, score_list))
	SCORES[combination] = dataset_score_list
	final_score = sum(
		score_list[-1][1] for score_list in dataset_score_list
	) / len(DATASET)
	print(f"Final score for {combination} -> {final_score}")

# Saving the scores.
with open(f"scores.{N}.pickle", "wb") as dest:
	pickle.dump(SCORES, dest)
