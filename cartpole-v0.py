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

		self.drop_out = nn.Dropout()

	def forward(self, x):

		out = self.fc1(x)
		out = self.fc2(out)
		out = self.drop_out(out)
		out = self.fc3(out)
		out = self.fc4(out)

		return out

class ReplayMemory:
	def __init__(self, Size):
		self.size = Size
		self.D = []

	def add(self, NMemory):
		self.D.append(NMemory)
		if (len(self.D) > self.size):
			self.D.pop(0)

	def sample(self, batch_size):
		return random.sample(self.D, batch_size)


model = NeuralN()
#ReplayM = ReplayMemory(1000)
α = 0.001 #Learning Rate

optimizer = torch.optim.Adam(model.parameters(), lr = α)
loss = nn.MSELoss()

state = env.reset()

ten = torch.from_numpy(state)
out = model(ten)
#print(state, out[0])
#print(float(out[0]),float(out[1]))
#print(out[0]>out[1], out)
#print(np.random.random_sample())
#print(out)
#out[0] = 0.8
#print(out)

#x = torch.randn(1, 3)
#y = torch.randn(1, 3)

#print(x)
#print(y)
#l = [[x,y],[x,y],[x,y],[x,y]]
#caten = [i[0] for i in l]
#print(caten)
#caten = torch.cat(caten,0)
#print(caten)

def train(env, ne):
	epsilon = 1
	gamma = 0.8
	e = 0
	memory = ReplayMemory(1000)
	mini_batch = 64

	#Episodes
	while (e < ne):
		state = env.reset()
		step = 0
		done = False
		rev = 0

		#Episode - Running till termination
		while not done:
			env.render()
			#Picking action by Epsilon-Greedy
			Q = model(torch.from_numpy(state))
			action = 0
			if Q[1]>Q[0]:
				action = 1
			action = np.random.randint(low = 0, high = 2, size = 1)[0] if np.random.random_sample() > epsilon else action

			epsilon = ((ne/10)/((ne/10) + e))

			#Execute action
			nstate, reward, done, _ = env.step(action)

			if done:
				reward = -1
			ten_reward = Q.detach().clone()
			ten_reward[action] = reward
			ten_Qnstate = torch.from_numpy(nstate)
			rev = rev + reward

			#print([torch.from_numpy(state), action, ten_reward, ten_Qnstate])
			#Storing Transition in the Replay Memory
			memory.add([torch.from_numpy(state), action, ten_reward, ten_Qnstate])
			#print(done)

			state = nstate

			#Sample random minibatch of transitions from Replay Memory
			if (len(memory.D) > mini_batch):
				batch = memory.sample(mini_batch)

				S_collection = [i[0] for i in batch]
				A_collection = [i[1] for i in batch]
				R_collection = [i[2] for i in batch]
				N_collection = [i[3] for i in batch]

				S_collection = torch.cat(S_collection, 0)
				R_collection = torch.cat(R_collection, 0)
				N_collection = torch.cat(N_collection, 0)

				S_collection = S_collection.reshape(mini_batch,4)
				R_collection = R_collection.reshape(mini_batch,2)
				N_collection = N_collection.reshape(mini_batch,4)

				Q = model(S_collection)
				QN = model(N_collection)
				BS = []
				Qmax = QN.max(1)
				for i in Qmax[0]:
					BS.append([i,i])
				Loss = loss(Q, (R_collection + gamma*QN))

				optimizer.zero_grad()
				Loss.backward()
				optimizer.step()
			step = step + 1
		e = e + 1
		print("Episode - ", e,"/",ne,"\tReward - ", rev)


def test(env, nt):
	t = 0
	reward_list = []
	model.eval()
	with torch.no_grad():
		while (t < nt):
			rev = 0
			state = env.reset()
			done = False
			while not done:
				env.render
				Q = model(torch.from_numpy(state))
				action = 0 if Q[0] > Q[1] else 1
				nstate, reward, done, _ = env.step(action)
				rev = rev + reward
				state = nstate
			reward_list.append(rev)
			t = t + 1
	return reward_list

train(env, 10)
#for param in (model.parameters()):
#	print(param.data)

print(test(env, 1))