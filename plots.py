import matplotlib.pyplot as plt
import random
import pickle

from common import N, M

with open(f"scores.{N}.pickle", "rb") as src:
	scores = pickle.load(src)

picks = random.sample(range(len(next(iter(scores.values())))), 5)

for pick in picks:
	plt.clf()
	for combination, score_tuples in scores.items():
		coefs, score_list = score_tuples[pick]
		plt.plot(range(M+1), score_list, label=combination)
	plt.legend()
	plt.xlabel("Iterations")
	plt.ylabel("Mean Squared Error")
	plt.savefig(f"plot.{pick}.png")
