import games
import numpy as np
import matplotlib.pyplot as plt
'''
Here, we will actually perform experiments on our games and generate plots.
'''

def experiment_binary_binary_decisions():
	# debugging purposes
	np.random.seed(4321)
	mw_vs_mw = True

	# Let's see how our weights move with different kinds of reward matrices.
	T = 50
	eta = 0.1

	reward_matrices = []
	game_titles = []

	# prisonner's dilemma
	reward_matrices += [[[[-5, -5], [0, -10]],
						[[-10, 0], [-1, -1]]]]
	game_titles += ["Prisonner's Dilemna"]

	# random
	reward_matrices += [[[[np.random.normal(i, 1) for j in range(2)] for i in range(2)]
						 for k in range(2)]]
	game_titles += ["Random Rewards"]
	print(reward_matrices[-1])

	# zero-sum
	reward_matrices += [[[[np.random.normal(i, 1) for j in range(2)] for i in range(2)]
						 for k in range(2)]]
	for i in range(2):
		for j in range(2):
			print(i, j)
			reward_matrices[2][i][j][1] = -1 * reward_matrices[2][i][j][0]
	game_titles += ["Zero Sum"]
	print(reward_matrices[-1])


	# mw vs mw
	if mw_vs_mw == True:
		for r, t in zip(reward_matrices, game_titles):
			weight_histories = []
			loss_histories = []
			for j in range(10):
				# Initialize MW algorithm
				weights = np.array([[1, 1], [1, 1]])
				weight_history = [[[v/sum(w) for v in w] for w in weights]]
				losses = np.array([[0, 0], [0, 0]])
				L = np.array([[0, 0], [0, 0]])
				actual_L = np.array([0., 0.])
				loss_history = [np.array(actual_L)]

				for epoch in range(T):
					game = games.binary_binary_decisions(r)
					pweights = [[v/sum(w) for v in w] for w in weights]
					moves = [np.random.choice(a=[0, 1], p=pweight) for pweight in pweights]
					# update losses
					losses[0][0] = game.play(0, moves[1])[0]
					losses[0][1] = game.play(1, moves[1])[0]
					losses[1][0] = game.play(moves[0], 0)[1]
					losses[1][1] = game.play(moves[0], 1)[1]
					L += losses
					actual_L += game.play(moves[0], moves[1])
					# update weights
					weights = np.array([np.exp(-1 * eta * L_i) for L_i in L])
					weight_history.append(pweights)
					loss_history.append(np.array(actual_L))

				weight_histories.append(weight_history)
				loss_histories.append(loss_history)

			weight_history = np.mean(weight_histories, axis=0)
			loss_history = np.mean(loss_histories, axis=0)
			plt.title(t)
			plt.xlabel("t")
			plt.plot(weight_history[:,0,0], label = "Player 1 Weight 1", color='green')
			plt.plot(weight_history[:,0,1], label = "Player 1 Weight 2", color='red')
			plt.plot(weight_history[:,1,0], label = "Player 2 Weight 1", color='orange')
			plt.plot(weight_history[:,1,1], label = "Player 2 Weight 2", color='blue')
			plt.legend()
			plt.show()

			plt.title(t + " Loss")
			plt.xlabel("t")
			plt.plot(loss_history[:,0], label = "Player 1 Losses", color='green')
			plt.plot(loss_history[:,1], label = "Player 2 Losses", color='red')
			plt.legend()
			plt.show()


def generate_random_blotto_expert():
	picks = np.random.choice([i for i in range(10)], 100)
	expert = [0 for i in range(10)]
	for val in picks:
		expert[val] += 1
	return expert


def blotto_game():
	# debugging purposes
	np.random.seed(4321)
	mw_vs_mw = True
	same_experts = True

	# Let's see how our weights move with different kinds of reward matrices.
	T = 1000
	eta = 0.1
	nexperts = 50
	if same_experts:
		experts = [generate_random_blotto_expert() for i in range(nexperts)]
		experts = [experts, experts]
	else:
		experts = [[generate_random_blotto_expert() for i in range(nexperts)] for j in range(2)]

	loss_histories = []
	for j in range(1):
		# Initialize MW algorithm
		weights = np.array([[0 for j in range(nexperts)] for i in range(2)])
		losses = np.array([[0 for j in range(nexperts)] for i in range(2)])
		L = np.array([[0 for j in range(nexperts)] for i in range(2)])
		actual_L = np.array([0., 0.])
		loss_history = [np.array(actual_L)]
		av_moves = []

		for epoch in range(T):
			game = games.blotto_game()
			pweights = [[v/sum(w) for v in w] for w in weights]
			moves = [np.random.choice(a=[i for i in range(nexperts)], p=pweight) for pweight in pweights]
			av_moves.append([experts[0][moves[0]], experts[1][moves[1]]])
			if epoch % 5 == 0:
				av_move = np.mean(av_moves, axis=0)
				av_moves = []
				plt.figure()
				plt.ylim((0, 30))
				plt.title("Average 5 Moves on Turn " + str(epoch))
				plt.bar([i+1-0.2 for i in range(10)], av_move[0], color='r', width=0.4, label='Player 1')
				plt.bar([i+1+0.2 for i in range(10)], av_move[1], color='b', width=0.4, label='Player 2')
				plt.legend()
				plt.xlabel("Battlfield")
				plt.ylabel("Soldiers")
				if same_experts:
					plt.savefig("graphs/gif/" + "blotto_same" + str(epoch))
				else:
					plt.savefig("graphs/gif/" + "blotto_diff" + str(epoch))
				plt.close()
			# update losses
			for i in range(nexperts):
				losses[0][i] = game.play(experts[0][i], experts[1][moves[1]])[0]
				losses[1][i] = game.play(experts[0][moves[0]], experts[1][i])[1]
			L += losses
			actual_L += game.play(experts[0][moves[0]], experts[1][moves[1]])
			# update weights
			weights = np.array([np.exp(-1 * eta * L_i) for L_i in L])
			loss_history.append(np.array(actual_L))
		loss_histories.append(loss_history)

	loss_history = np.mean(loss_histories, axis=0)
	plt.title("Blotto MW vs MW")
	plt.plot(loss_history[:,0], label='Player 1 Losses')
	plt.plot(loss_history[:,1], label='Player 2 Losses')
	plt.xlabel("t")
	plt.legend()
	plt.show()
	






if __name__ == '__main__':
	blotto_game()





