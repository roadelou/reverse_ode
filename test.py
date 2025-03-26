import torch
import random
from tabulate import tabulate

from model import *

# Loading the model from disk.
MODEL.load_state_dict(torch.load("model.pth", weights_only=True))
MODEL.eval()

# Loading the data.
_, dataset = load_data("dataset.pickle")

# Some trials so I can see for myself.
samples = random.sample(dataset, 5)
for label, coefs, data in samples:
	print()
	with torch.no_grad():
		prediction = MODEL(coefs, data)
	print(tabulate([["REF", "SRC", "PRED"]] + [
		[ref, src, pred]
		for ref, src, pred in zip(label, coefs, prediction)
	]))
