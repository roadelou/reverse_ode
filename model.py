import torch
from torch import nn
import pickle
import numpy as np
from tqdm import tqdm	# For progress bar.

from common import N

# MODEL = nn.Sequential(
# 	nn.LogSigmoid(),
# 	nn.Linear(2*1000 + 2*N, 256),
# 	nn.LogSigmoid(),
# 	nn.Linear(256, 64),
# 	nn.LogSigmoid(),
# 	nn.Linear(64, 32),
# 	nn.LogSigmoid(),
# 	nn.Linear(32, 2*N),
# 	nn.LogSigmoid(),
# )

class ODEFinder(nn.Module):
	def __init__(self, n):
		super().__init__()
		self.conv_stack = nn.Sequential(
			nn.Conv1d(2, 4*n, n),
			nn.AvgPool1d(n),
			nn.Conv1d(4*n, 4*n, n),
			nn.AvgPool1d(n),
			nn.Conv1d(4*n, 4*n, 2*n),
			nn.AvgPool1d(2*n),
			nn.Flatten(0, -1)
		)
		self.dense_reduc = nn.Sequential(
			# nn.Linear(150, 2*n),	# 3
			# nn.Linear( 70, 2*n),	# 5
			nn.Linear(492, 2*n),	# 2
			nn.ReLU()
		)
		self.dense_final = nn.Linear(4*n, 2*n)
	
	def forward(self, coefs, values):
		features = self.conv_stack(values)
		reduc_input = torch.cat((coefs, features), 0)
		reduc_output = self.dense_reduc(reduc_input)
		final_input = torch.cat((coefs, reduc_output), 0)
		return self.dense_final(final_input)

MODEL = ODEFinder(N)

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
			torch.from_numpy(error_coefs),
			torch.from_numpy(np.stack((
				ref_val,
				error_val
			)))
		)
		for ref_coefs, error_coefs, ref_val, error_val
		in raw_dataset
	]
	#
	# Converting the data to float for later.
	float_tensors = [
		(label.to(torch.float), coefs.to(torch.float), data.to(torch.float))
		for label, coefs, data in double_tensors
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
	for label, coefs, data in tqdm(dataset):
		#
		# We predict the coefficients.
		prediction = model(coefs, data)
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
		for label, coefs, data in dataset:
			#
			# We predict the coefficients.
			prediction = model(coefs, data)
			#
			# We compute the accuracy of the model.
			loss += loss_fn(prediction, label).item()
		return loss / len(dataset)
