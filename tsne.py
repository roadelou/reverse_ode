from sklearn.decomposition import PCA
import pickle
import numpy as np
import matplotlib.pyplot as plt

from common import N

# Initial dataset
print("LOADING")
with open(f"dataset.{N}.pickle", "rb") as src:
	DATASET = pickle.load(src)

# Loading the scores
with open(f"scores.{N}.pickle", "rb") as src:
	SCORES = pickle.load(src)

# We train the PCA on all of the samples in the dataset. This will ensure we
# we have somewhat stable plots.
print("TRAINING")
# pca = PCA(n_components=2)

class FakePCA:
	def transform(self, X):
		return X[:, :2]
	
	def fit_transform(self, X):
		return self.transform(X)

pca = FakePCA()

X_train = np.array([
	ref_coefs
	for ref_coefs, _, _, _ in DATASET
])
Y_train = pca.fit_transform(X_train)

# We plot the points on a graph.
print("PLOTTING")
plt.clf()
plt.scatter(*zip(*Y_train))
plt.title("Full dataset PCA")
plt.savefig(f"pca.full.{N}.png")

# For each method, we plot the score for points everywhere in the space.
for combination, score_tuples in SCORES.items():
	print(combination)
	coefs, score_lists = zip(*score_tuples)
	final_scores = [score_list[1] for score_list in score_lists]
	print(final_scores[0])
	print(np.mean(final_scores))
	projected_coefs = pca.transform(np.array(coefs))
	X, Y = zip(*projected_coefs)
	plt.clf()
	plt.scatter(X, Y, c=final_scores)
	plt.colorbar()
	plt.clim(0, 1)
	title = combination
	if combination == "lin/zero":
		title = "Linear Model"
	elif combination == "nn/zero":
		title = "Simple NN"
	elif combination == "nn/nn":
		title = "Simple NN + Reinforcement Learning"
	plt.title(title)
	plt.savefig(f"pca.{combination.replace('/','.')}.{N}.png")
