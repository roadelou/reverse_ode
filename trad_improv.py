import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import (
	LinearRegression,
	Perceptron,
	ElasticNet,
	Lars
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
	RandomForestRegressor, HistGradientBoostingRegressor
)
from common import N

def load_data(path):
	with open(path, "rb") as src:
		raw_dataset = pickle.load(src)
	ref_coefs, error_coefs, ref_vals, error_vals = zip(*raw_dataset)
	#
	# Returning numpy arrays with the whole data.
	return (
		np.array(ref_coefs),
		np.array(error_coefs),
		np.array(ref_vals),
		np.array(error_vals)
	)

# GOOD
MODEL = LinearRegression()
MODEL_NAME = "linear"

# Out of memory
# MODEL = KernelRidge()
# MODEL_NAME = "kernel_ridge"

# Only 1d
# MODEL = SVR()
# MODEL_NAME = "svr"

# Only 1d
# MODEL = LinearSVR()
# MODEL_NAME = "linear_svr"

# GOOD
# MODEL = KNeighborsRegressor()
# MODEL_NAME = "kneighbors"

# Out of memory
# MODEL = GaussianProcessRegressor()
# MODEL_NAME = "gaussian_process"

# GOOD
# MODEL = PLSRegression(n_components=2 * 3)
# MODEL_NAME = "pls_regression"

# VERY SLOW
# MODEL = DecisionTreeRegressor()
# MODEL_NAME = "tree"

# TOO SLOW
# MODEL = RandomForestRegressor(n_jobs=-1)
# MODEL_NAME = "random_forest"

# 1d only
# MODEL = HistGradientBoostingRegressor()
# MODEL_NAME = "hgboost"

# 1d only
# MODEL = Perceptron()
# MODEL_NAME = "perceptron"

# GOOD
# MODEL = ElasticNet()
# MODEL_NAME = "elasticnet"

# GOOD
# MODEL = Lars()
# MODEL_NAME = "lars"

print("LOADING DATA")
ref_coefs, error_coefs, ref_vals, error_vals = load_data(f"dataset.{N}.pickle")
X_train, X_test, Y_train, Y_test = train_test_split(
	ref_vals - error_vals, ref_coefs - error_coefs, test_size=0.1
)

# Training
print("TRAINING")
MODEL.fit(X_train, Y_train)

# Testing
print("TESTING")
Y_pred = MODEL.predict(X_test)

print("MEASURING")
# Reference error.
Y_zeros = np.zeros_like(Y_pred)
ref_error = mean_squared_error(Y_test, Y_zeros)
print(f"Reference mean squared error: {ref_error}")

# Computing error
error = mean_squared_error(Y_test, Y_pred)
print(f"Model mean squared error: {error}")

# SAVING MODEL
print("SAVING MODEL")
with open(f"improve.{MODEL_NAME}.{N}.pickle", "wb") as dest:
	pickle.dump(MODEL, dest)
