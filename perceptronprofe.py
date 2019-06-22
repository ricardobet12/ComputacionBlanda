import numpy as np

class Perceptron(object):

	def __init__(self, eta=0.01, epochs=50):
		self.eta = eta
		self.epochs = epochs
		
	def train(self, X, y):
	
		self.w_ = np.zeros(1 + X.shape[1])
		self.errors_ = []
		
		for _ in range(self.epochs):
			errors = 0
			for xi, target in zip(X, y):
				update = self.eta * (target - self.predict(xi))
				self.w_[1:] +=  update * xi
				self.w_[0]  +=  update
				errors += int(update != 0.0)
			self.errors_.append(errors)
		return self
		
	def net_input(self, X):
		return np.dot(X, self.w_[1:]) + self.w_[0]
		
	def predict(self, X):
		return np.where(self.net_input(X) >= 0.0, 1, -1)

class AdalineGD(object):

	def __init__(self, eta=0.01, epochs=50):
		self.eta = eta
		self.epochs = epochs
		
	def train(self, X, y):
	
		self.w_ = np.zeros(1 + X.shape[1])
		self.cost_ = []
		
		for i in range(self.epochs):
			output = self.net_input(X)
			errors = (y - output)
			self.w_[1:] += self.eta * X.T.dot(errors)
			self.w_[0] += self.eta * errors.sum()
			cost = (errors**2).sum() / 2.0
			self.cost_.append(cost)
		return self
		
	def net_input(self, X):
		return np.dot(X, self.w_[1:]) + self.w_[0]
		
	def activation(self, X):
		return self.net_input(X)
		
	def predict(self, X):
		return np.where(self.activation(X) >= 0.0, 1, -1)