import torch
import numpy as np
from torch.autograd import Variable


class RidgeRegression(torch.nn.Module):
	def __init__(self, input_dim, reg_lambda, dtype="float"):
		super(RidgeRegression, self).__init__()
		self.input_dim = input_dim
		self.reg_lambda = reg_lambda
		self.linear = torch.nn.Linear(self.input_dim, out_features=1)
		self.criterion = torch.nn.MSELoss(size_average=True)
		self.dtype = dtype
		if self.dtype == "double":
			for w in self.parameters():
				w.data = w.data.type(torch.DoubleTensor)

	def forward(self, x, y):
		self.output = self.linear(x)
		self.loss = self.criterion(self.output, y)
		return self.loss

	def predict(self, x):
		output = self.linear(x)
		pred = output.data.cpu().numpy()
		return pred


def ridge_regression_grad_test():
	n_sample = 4
	n_dim = 3
	X = Variable(torch.DoubleTensor(np.random.normal(size=(n_sample, n_dim) ) ) )
	Y = Variable(torch.DoubleTensor(np.random.normal(size=(n_sample) ) ) )
	regressor = RidgeRegression(input_dim=n_dim, reg_lambda=np.random.normal(), dtype="double")
	loss1 = regressor.forward(X, Y)
	loss_diff = 0.0
	move = 1e-9
	loss1.backward()
	for w in regressor.parameters():
		loss_diff += torch.sum(w.grad.data * move)
	for w in regressor.parameters():
		w.data += move
	loss2 = regressor.forward(X, Y)
	assert np.abs((loss2[0] - loss1[0] ).data.cpu().numpy() - loss_diff) < 1e-6
	# print("loss finite diff ", loss2[0] - loss1[0], " projected loss change ", loss_diff)
	print("Ridge regression gradient test done!")


if __name__ == "__main__":
	ridge_regression_grad_test()		







