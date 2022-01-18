import gym
import torch
import torchvision
import torch.nn as nn
import random
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
		self.ReLU = nn.ReLU()

		self.drop_out = nn.Dropout()

	def forward(self, x):

		out = self.fc1(x)
		out = self.ReLU(out)
		out = self.fc2(out)
		out = self.ReLU(out)
		#out = self.drop_out(out)
		out = self.fc3(out)
		out = self.ReLU(out)
		out = self.fc4(out)
		#out = self.ReLU(out)

		return out


def test(env, nt, show):
	t = 0
	reward_list = []
	model_targe.eval()
	with torch.no_grad():
		while (t < nt):
			rev = 0
			state = env.reset()
			done = False
			while not done:
				if show:
					env.render()
				Q = model_targe(torch.from_numpy(state))
				action = 0 if Q[0] > Q[1] else 1
				#print(action)
				nstate, reward, done, _ = env.step(action)
				rev = rev + reward
				state = nstate
			reward_list.append(rev)
			t = t + 1
	return reward_list


model_targe = NeuralN()
#model_targe.load_state_dict(model_learn.state_dict())
model_targe.load_state_dict(torch.load("saved_parameters1.pt"))
#model_targe = torch.load("saved_parameters1.pt")
model_targe.eval()

state = env.reset()
rew = (test(env, 100, 0))

x = [i for i in range(len(rew))]

plt.plot(x, rew, label = "Reward")
print(rew)
test(env, 20, 1)
plt.show()