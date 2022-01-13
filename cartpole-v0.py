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
		self.ReLU = nn.ReLU()

		self.drop_out = nn.Dropout()

	def forward(self, x):

		out = self.fc1(x)
		#out = self.ReLU(out)
		out = self.fc2(out)
		#out = self.ReLU(out)
		#out = self.drop_out(out)
		out = self.fc3(out)
		#out = self.ReLU(out)
		out = self.fc4(out)
		#out = self.ReLU(out)

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

def train(env, ne):
	epsilon = 1
	gamma = 0.9
	e = 0
	memory = ReplayMemory(1000)
	mini_batch = 128
	rev_list = []

	#Episodes
	while (e < ne):
		state = env.reset()
		step = 0
		done = False
		rev = 0
		count = 0

		#Episode - Running till termination
		while not done:
			#env.render()
			#Picking action by Epsilon-Greedy
			Q = []
			with torch.no_grad():
				Q = model_learn(torch.from_numpy(state))
			action = 0 if Q[0] > Q[1] else 1
			action = np.random.randint(low = 0, high = 2, size = 1)[0] if np.random.random_sample() > epsilon else action

			epsilon = ((ne/10)/((ne/10) + e))
			#if (e > 0.9*ne):
			#	epsilon = 0.1

			#Execute action
			nstate, reward, done, _ = env.step(action)

			if done:
				reward = -1
			ten_reward = Q.detach().clone()
			ten_reward[action] = reward
			ten_reward[1-action] = 0
			ten_Qnstate = torch.from_numpy(nstate)
			rev = rev + reward

			#print([torch.from_numpy(state), action, ten_reward, ten_Qnstate])
			#Storing Transition in the Replay Memory
			memory.add([torch.from_numpy(state), action, ten_reward, ten_Qnstate, done])
			#print(done)
			state = nstate

			#Training the model on a sample of experience
			if (len(memory.D) > mini_batch):
				batch = memory.sample(mini_batch)

				S_collection = [i[0] for i in batch]
				#A_collection = [i[1] for i in batch]
				R_collection = [i[2] for i in batch]
				N_collection = [i[3] for i in batch]

				S_collection = torch.cat(S_collection, 0)
				R_collection = torch.cat(R_collection, 0)
				N_collection = torch.cat(N_collection, 0)

				S_collection = S_collection.reshape(mini_batch,4)
				R_collection = R_collection.reshape(mini_batch,2)
				N_collection = N_collection.reshape(mini_batch,4)

				#print(R_collection[0])

				#Q Values of the actions in current state
				Q = model_learn(S_collection)
				#Q Values of the actions in next state
				QN = []
				with torch.no_grad():
					QN = model_targe(N_collection)
				for i in range(mini_batch):
					if batch[i][4]:
						QN[i][0], QN[i][1] = 0, 0
						#print(QN[i], batch[i])

				#Picking the Max Value as the Q value for the next state
				BS = []
				Qmax = QN.max(1)
				for i in Qmax[0]:
					BS.append([i,i])
				QN = torch.FloatTensor(BS)
				#print(QN)

				targe = R_collection + gamma*QN
				#print(Q[0], R_collection[0], QN[0], targe[0], "\n")
				#print(targe)

				#print(R_collection,QN)
				#print(QN)
				Loss = loss(input = Q, target = targe)
				#print(Loss)
				optimizer.zero_grad()
				Loss.backward()
				optimizer.step()
				if (count == 500):
					model_targe.load_state_dict(model_learn.state_dict())
					count = 0
				else:
					count = count + 1
			step = step + 1
		e = e + 1
		print("Episode - ", e,"/",ne,"\tReward - ", rev)
		rev = test(env, 1, 0)
		rev_list.append(rev)
	return rev_list


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

model_learn = NeuralN()
model_targe = NeuralN()
model_targe.load_state_dict(model_learn.state_dict())
model_targe.eval()

#s = [0.0677,0.0197,-0.2054,-0.5108]
#s = torch.FloatTensor(s)
#print(model_learn(s), model_targe(s))

#ReplayM = ReplayMemory(1000)
α = 0.0001 #Learning Rate

optimizer = torch.optim.Adam(model_learn.parameters(), lr = α)

loss = nn.MSELoss()
#loss = nn.CrossEntropyLoss()

state = env.reset()

#for param in (model.parameters()):
#	print(param.data)

r = train(env, 1000)
x = [i for i in range(len(r))]
plt.plot(x, r)

#for param in (model.parameters()):
#	print(param.data)

print(test(env, 10, 1))

#plt.show()