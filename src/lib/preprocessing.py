import numpy as np
import pandas as pd
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

def get_reduced_dev_split(train_data, dev_data):
    rows_to_transfer = len(dev_data) - 1000
    transfer_data = dev_data.iloc[:rows_to_transfer]
    dev_data = dev_data.iloc[rows_to_transfer:]
    train_data = pd.concat([train_data, transfer_data])
    print(f'Train size: {len(train_data)}')
    print(f'Dev size: {len(dev_data)}')
	
    return train_data, dev_data