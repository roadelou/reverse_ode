import torchviz
import pickle
import torch
import numpy as np

from oneshot import MODEL as ONESHOT
from model import MODEL

from common import N

with open(f"dataset.{N}.pickle", "rb") as src:
	DATASET = pickle.load(src)

ref_coefs, error_coefs, ref_val, error_val = DATASET[0]

oneshot_res = ONESHOT(torch.from_numpy(ref_val).to(torch.float))

torchviz.make_dot(
	oneshot_res,
	params=dict(ONESHOT.named_parameters()),
).render(f"oneshot.{N}", format="png")

model_res = MODEL(
	torch.from_numpy(error_coefs).to(torch.float),
	torch.from_numpy(np.stack((
		ref_val,
		error_val
	))).to(torch.float)
)

torchviz.make_dot(
	model_res,
	params=dict(MODEL.named_parameters()),
).render(f"model.{N}", format="png")
