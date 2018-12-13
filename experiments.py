import games
import numpy as np
import matplotlib.pyplot as plt
'''
Here, we will actually perform experiments on our games and generate plots.
'''

def binary_decisions_mw():
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


def blotto_game_mw():
	# debugging purposes
	np.random.seed(4321)
	mw_vs_mw = True
	same_experts = False

	T = 2000
	nexperts = 10
	eta = (np.log(nexperts)/T) ** 0.5
	if same_experts:
		experts = [generate_random_blotto_expert() for i in range(nexperts)]
		experts = [experts, experts]
	else:
		experts = [[generate_random_blotto_expert() for i in range(nexperts)] for j in range(2)]

	loss_histories = []
	for j in range(1):
		# Initialize MW algorithm
		weights = np.array([[1 for j in range(nexperts)] for i in range(2)])
		losses = np.array([[0 for j in range(nexperts)] for i in range(2)])
		L = np.array([[0 for j in range(nexperts)] for i in range(2)])
		actual_L = np.array([0., 0.])
		loss_history = [np.array(actual_L)]
		av_weights = []

		for epoch in range(T):
			game = games.blotto_game()
			pweights = [[v/sum(w) for v in w] for w in weights]
			moves = [np.random.choice(a=[i for i in range(nexperts)], p=pweight) for pweight in pweights]
			av_weights.append(pweights)
			if epoch % 10 == 0:
				av_weight = np.mean(av_weights, axis=0)
				av_weights = []
				plt.figure()
				plt.ylim((0, 1))
				plt.title("Average Weights (last 10) on Turn " + str(epoch))
				plt.bar([i+1-0.2 for i in range(nexperts)], av_weight[0], color='r', width=0.4, label='Player 1')
				plt.bar([i+1+0.2 for i in range(nexperts)], av_weight[1], color='b', width=0.4, label='Player 2')
				plt.legend()
				plt.xlabel("Expert Weights")
				plt.ylabel("Soldiers")
				if same_experts:
					plt.savefig("graphs/gif/" + "blotto_same_" + str(nexperts) + "/" + str(epoch))
				else:
					plt.savefig("graphs/gif/" + "blotto_diff_" + str(nexperts) + "/" + str(epoch))
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

def blotto_game_adahedge():
    # debugging purposes
    np.random.seed(4321)
    hedge_vs_hedge = False
    same_experts = True
    
    T = 2000
    nexperts = 10
    eta = (np.log(nexperts)/T) ** 0.5
    eta_hedge = [2, 2]
    if hedge_vs_hedge == False:
        eta_hedge[1] = eta
    eta_hedge_history = [eta_hedge]

    if same_experts:
        experts = [generate_random_blotto_expert() for i in range(nexperts)]
        experts = [experts, experts]
    else:
        experts = [[generate_random_blotto_expert() for i in range(nexperts)] for j in range(2)]

    # Initialize MW algorithm
    weights = np.array([[1/nexperts for j in range(nexperts)] for i in range(2)])
    losses = np.array([[0. for j in range(nexperts)] for i in range(2)])
    L = np.array([[0. for j in range(nexperts)] for i in range(2)])
    actual_L = np.array([0., 0.])
    loss_history = [np.array(actual_L)]
    av_weights = []
    hedge_val = [0, 0]

    path = "adahedge_vs_adahedge"
    if hedge_vs_hedge == False:
        path = "adahedge_vs_mw"
    if same_experts:
        path = path + "_same_"
    else:
        path = path + "_diff_"
    path = path + str(nexperts) + "/"
    path = "graphs/gif/" + path

    for epoch in range(T):
        game = games.blotto_game()
        pweights = [[v/sum(w) for v in w] for w in weights]
        moves = [np.random.choice(a=[i for i in range(nexperts)], p=pweight) for pweight in pweights]
        av_weights.append(pweights)
        if epoch % 10 == 0:
            av_weight = np.mean(av_weights, axis=0)
            av_weights = []
            plt.figure()
            plt.ylim((0, 1))
            plt.title("Average Weights (last 10) on Turn " + str(epoch))
            plt.bar([i+1-0.2 for i in range(nexperts)], av_weight[0], color='r', width=0.4, label='Player 1')
            plt.bar([i+1+0.2 for i in range(nexperts)], av_weight[1], color='b', width=0.4, label='Player 2')
            plt.legend()
            plt.xlabel("Expert Weights")
            plt.ylabel("Soldiers")
            type = "adahedge_vs_adahedge"
            if hedge_vs_hedge == False:
                type = "adahedge_vs_mw"
            if same_experts:
                plt.savefig(path + str(epoch))
            else:
                plt.savefig(path + str(epoch))
            plt.close()
        # update losses
        for i in range(nexperts):
            losses[0][i] = game.play(experts[0][i], experts[1][moves[1]])[0]
            losses[1][i] = game.play(experts[0][moves[0]], experts[1][i])[1]
        L += losses
        actual_L += game.play(experts[0][moves[0]], experts[1][moves[1]])
        for i in range(2):
            hedge_val[i] += weights[i].dot(losses[i]) - (-1 / eta_hedge[i]) * np.log(weights[i].dot(np.exp(-1 * eta_hedge[i] * losses[i])))
        # update weights
        #weights[0] = np.exp(-1 * eta_hedge[0] * L[0])
        #weights[1] = np.exp(-1 * eta_hedge[1] * L[1])
        for i in range(2):
            weights[i] = weights[i] * np.exp(-1 * eta_hedge[i] * losses[i]) / weights[i].dot(np.exp(-1 * eta_hedge[i] * losses[i]))
        loss_history.append(np.array(actual_L))
        for i in range(2):
            if hedge_vs_hedge == False and i == 1:
                break
            if hedge_val[i] > np.log(nexperts) * (1 / eta_hedge[i] + 1 / (np.e - 1)):
                hedge_val[i] = 0
                eta_hedge[i] /= 2
                weights[i] = [1/nexperts for i in range(nexperts)]
                L[i] = [0 for i in range(nexperts)]

        eta_hedge_history.append(np.array(eta_hedge))
            
    plot_title = "Blotto MW vs AdaHedge"
    if hedge_vs_hedge:
        plot_title = "Blotto AdaHedge vs AdaHedge"

    loss_history = np.array(loss_history)
    plt.figure()
    plt.title(plot_title)
    plt.plot(loss_history[:,0], label='Player 1 Losses')
    plt.plot(loss_history[:,1], label='Player 2 Losses')
    plt.xlabel("t")
    plt.legend()
    plt.savefig(path + "loss_plot")
    plt.show()
    plt.close()

    eta_hedge_history = np.array(eta_hedge_history)
    plt.figure()
    plt.title(plot_title)
    plt.plot(np.log(eta_hedge_history[:,0]), label="Player 1 Learning Rate")
    if hedge_vs_hedge:
        plt.plot(np.log(eta_hedge_history[:,1]), label="Player 2 Learning Rate")
        plt.legend()
    plt.xlabel("t")
    plt.ylabel("Ln of AdaHedge Learning Rate")
    plt.savefig(path + "learning_rate_plot")
    plt.show()
    plt.close()


if __name__ == '__main__':
    #binary_decisions_mw()
    blotto_game_mw()
    #blotto_game_adahedge()





