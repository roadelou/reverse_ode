import ode
from tqdm import tqdm	# For progress bar.
import pickle

from common import N

dataset = [
	ode.draw_sample(N, max_coefs=1, max_eps=0.1)
	for _ in tqdm(range(100000))
]

with open(f"dataset.{N}.pickle", "wb") as dest:
	pickle.dump(dataset, dest)
