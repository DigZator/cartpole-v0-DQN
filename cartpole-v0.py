# A pole is attached by an un-actuated joint to a cart,
# which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart.
# The pendulum starts upright, and the goal is to prevent it from falling over.
# A reward of +1 is provided for every timestep that the pole remains upright.
# The episode ends when the pole is more than 15 degrees from vertical,
# or the cart moves more than 2.4 units from the center.

import gym
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('CartPole-v0')

class NeuralN(nn.Module):
	def __init__(self):
		super(NeuralN,self).__init__()

		self.fc1 = nn.Linear(in_features = 4, out_features = 8)
		self.fc2 = nn.Linear(in_features = 8, out_features = 8)
		self.fc3 = nn.Linear(in_features = 8, out_features = 4)
		self.fc4 = nn.Linear(in_features = 4, out_features = 2)

		self.drop_out = nn.Dropout()

		self.criterion = nn.CrossEntropyLoss()

	def forward(self, x):

		out = self.fc1(out)
		out = self.fc2(out)
		out = self.drop_out(out)
		out = self.fc3(out)
		out = self.fc4(out)

		return out

	def criterion(self, ap, ac):
		loss = nn.criterion(ap, ac)
		return loss

