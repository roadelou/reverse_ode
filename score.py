import pickle

from common import N

# Loading the scores
print("LOADING")
with open(f"scores.{N}.pickle", "rb") as src:
	SCORES = pickle.load(src)

for combination, dataset_score_list in SCORES.items():
	print(f"Testing the combination: {combination}")
	final_score = sum(
		score_list[-1][1] for score_list in dataset_score_list
	) / 2000
	print(f"Final score for {combination} -> {final_score}")
