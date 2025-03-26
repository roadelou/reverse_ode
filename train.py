import torch

from model import *
from common import N

# Number of epochs.
EPOCHS = 5

# Loading the data
train_dataset, test_dataset = load_data(f"dataset.{N}.pickle")

# Computing the reference loss if the model does nothing.
ref_loss = 0
for label, coefs, _ in test_dataset:
	ref_loss += LOSS_FN(coefs, label).item()
avg_ref_loss = ref_loss / len(test_dataset)
print(f"Reference loss: {avg_ref_loss}")

for _ in range(EPOCHS):
	train(train_dataset, MODEL, LOSS_FN, OPTIMIZER)
	avg_loss = test(test_dataset, MODEL, LOSS_FN)
	print(f"Average loss: {avg_loss}")

torch.save(MODEL.state_dict(), f"improv.{N}.pth")
