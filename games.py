'''
Comprehensive File for games.
'''

import numpy as np



'''
Basic game. cart_search consists of two goals randomly generated
on the cartesian plane with X and Y iid ~ U[-50, 50]. After covariance
shift, we will have X and Y iid ~ U[-200, 200].

The state space consists of current location (x, y). We start at the
point equidistant between the two goals, and our action space consists
of (d, theta). d is the distance to travel, and theta is the angle w.r.t.
the x-axis to travel. d is continuous between [-5, 5] and theta is continuous 
between [0, 2pi]. We update location (x, y) by:

x <- x + d * cos(theta)
y <- y + d * sin(theta)

Loss is given by how the cartesian distance from the goal.

We have two versions of this game - deterministic and noisy.
We try amplified noise - multiply d and theta by
iid Z ~ Gauss(mu, sigma). 

Given the difficulty of precisely reaching the goal, we apply a time horizon T
and compare different algorithms by their final loss. With the way we 
have constructed our reward function, this can be viewed as whichever algorithm
came closer to the goal within the time horizon.

This is classified as a zero-sum game. 
'''

class cart_search:
	
	def __init__(self, goal_state1, goal_state2, noisy, mu, sigma):
		self.goal = (goal_state1, goal_state2)
		self.state = ((goal_state[0][0] + goal_state[1][0]) / 2, (goal_state[0][1] + goal_state[1][1]) / 2)
		self.noisy = noisy
		self.mu = mu
		self.sigma = sigma

	def get_state(self):
		return [self.goal[0][0], self.goal[0][1], self.goal[1][0],
		 self.goal[1][1], self.state[0], self.state[1]]

	# make a move, returns reward
	def move(self, d, theta, player):
		# if invalid move, we do nothing
		if d < -5 or d > 5 or theta < 0 or theta > 2 * np.pi or (player != 0 and player != 1):
			return -1000

		# if noisy, apply noise
		if self.noisy:
			d *= np.random.normal(self.mu, self.sigma)
			theta *= np.random.normal(self.mu, self.sigma)

		# move
		oldstate = self.state
		self.state = (self.state[0] + d * np.cos(theta),
					  self.state[1] + d * np.sin(theta))

		# calculate and return loss
		return cart_search.dist(self.state, self.goal[player])

	# get cartesian distance between two states
	def dist(s1, s2):
		return ((s1[0] - s2[0]) ** 2 + (s1[1] - s2[1]) ** 2) ** 0.5




'''
Another simple game. Re_zero starts with some random number, U[25, 50]. After
covariance shift, we have U[50, 100]. Players take turns subtracting {1, 2, 3} from
this number. Whichever player hits a nonpositive number first loses and incurs -100 loss.
'''

class re_zero:
	def __init__(self, start):
		self.value = start

	def get_state(self):
		return [self.value]

	def play(self, move):
		if self.value <= 0 or (move != 1 and move != 2 and move != 3):
			return -1000

		self.value -= move

		if self.value <= 0:
			return -10



'''
Very simple game. binary_binary_decisions is a non-zero-sum game revolving around the two players, 
both making one move. We have an array of rewards r[2][2][2], where player i 
makes choice c_i, and the reward for player i is r[c_0][c_1][i]. 

Note: We can set up r such that this game becomes zero-sum. We will do this with Prisonner's Dilemma.
'''

class binary_binary_decisions:

	def __init__(self, r):
		self.r = r

	def get_state(self):
		return self.r

	def play(self, move1, move2):
		return -1 * np.array(self.r[move1][move2])



'''
Blotto Game. Zero-sum game where each player has 100 soldiers, and there are 10 battlefields.
They divide up their soldiers among the battlefields. You win a battlefield if you have
more soldiers. The player with the most won battlefields wins in the end. We change this up
by instead of having players send all their soldiers all at once, they send them one battlefield
at a time. The player that loses the battlefield incurs loss 1, while the winner incurs no loss.

Possible covariance shift: instead of 10 battlefields, now there are 20. 
'''

class blotto_game:

	def __init__(self):
		self.soldiers = [100, 100]

	def get_state(self):
		return [self.games] + self.soldiers

	def play(self, move1, move2):
		if sum(move1) > 100 or sum(move2) > 100:
			return

		tmp = 0
		for i in range(10):
			if move1[i] > move2[i]:
				tmp += 1
			elif move1[i] < move2[i]:
				tmp += -1
		
		return [-1 * tmp, tmp]

