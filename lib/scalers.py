import numpy as np
from sklearn.preprocessing import StandardScaler

class JointScaler():
	def __init__(self):
		self.means = None
		self.stddev = None

	def fit_transform(self, data):
		self.fit(data)
		return self.transform(data)
	
	def fit(self, data):
		self.means = np.mean(data, axis=0)
		centereddata = data - self.means
		self.stddev = np.std(centereddata)

	def transform(self, data):
		return (data - self.means) / self.stddev

	def inverse_transform(self, data):
		return (data * self.stddev) + self.means
	

scalers = {
	'independent': StandardScaler,
	'joint': JointScaler
}