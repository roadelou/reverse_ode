import torch
from torch import nn
import pickle
import numpy as np
from tqdm import tqdm	# For progress bar.

from common import N

MODEL = nn.Sequential(
	nn.LogSigmoid(),
	nn.Linear(1000, 512),
	nn.LogSigmoid(),
	nn.Linear(512, 256),
	nn.LogSigmoid(),
	nn.Linear(256, 64),
	nn.LogSigmoid(),
	nn.Linear(64, 2*N),
	nn.LogSigmoid(),
)

LOSS_FN = nn.MSELoss()
OPTIMIZER = torch.optim.SGD(MODEL.parameters(), lr=1e-1)

def load_data(path):
	with open(path, "rb") as src:
		raw_dataset = pickle.load(src)
	#
	# We format the data for pytorch.
	double_tensors = [
		(
			torch.from_numpy(ref_coefs),
			torch.from_numpy(ref_val)
		)
		for ref_coefs, _, ref_val, _
		in raw_dataset
	]
	#
	# Converting the data to float for later.
	float_tensors = [
		(label.to(torch.float), data.to(torch.float))
		for label, data in double_tensors
	]
	#
	# We pick 10% of the data to go in testing dataset.
	len_test = len(float_tensors) // 10
	#
	# Returning the tensors.
	return float_tensors[:-len_test], float_tensors[-len_test:]

def train(dataset, model, loss_fn, optimizer):
	#
	# We train the model.
	model.train()
	for label, data in tqdm(dataset):
		#
		# We predict the coefficients.
		prediction = model(data)
		#
		# We compute the accuracy of the model.
		loss = loss_fn(prediction, label)
		#
		# Backpropagation.
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

def test(dataset, model, loss_fn):
	# Testing the model.
	model.eval()
	loss = 0
	with torch.no_grad():
		for label, data in dataset:
			#
			# We predict the coefficients.
			prediction = model(data)
			#
			# We compute the accuracy of the model.
			loss += loss_fn(prediction, label).item()
		return loss / len(dataset)
