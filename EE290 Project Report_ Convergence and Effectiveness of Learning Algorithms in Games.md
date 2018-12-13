
# EE290 Project Report: Convergence and Effectiveness of Learning Algorithms in Games

## Problem setting

In this project, we will consider some very simple two-player games that are zero-sum OR non-zero-sum, where the opposing players do not know their opponent's utility funtions. We are interested in the following questions:



1. What happens if the players play ****no-regret**** algorithms against each other? Do we converge to ****Nash/Correlated equilibrium****?

2. How does the ****learning rate**** change in ****Adaptive Hedge**** vs no-regret algorithms? What about Adahedge vs itself? What about convergence overall?

3. How well can these algorithms adapt to a ****covariance shift****, and how useful is ****batch normalization**** in minimizing regret after a covariance shift?



## Introduction



### Environment

In class, we often interpreted online learning from the context of predicting a sequence. In this project, we follow the same framework - we have a sequence of the same game, with the two players learning as they play each other more and more.



### No-Regret Algorithms

Regret is the measure of how well we did, compared against how well we could have done. It is calculated post-experiment - we look over all the actions we took, and see if we could have chosen the same action each time that would have resulted in a lower loss. Cumulative regret is the sum total of the differences in potential and actual losses over each action. Typically, with a time horizon T, regret scales with some order of T.



We define no-regret algorithms as those that have *_average_* regret converge to zero as the time horizon goes to infinity. More formally, an algorithm is *_no-regret_* if:


$\underset{T \rightarrow \infty}{lim} R(f)/T = 0$



Notice, then, that if cumulative regret scales sublinear to T, we would have a no-regret algorithm. From this point onwards, we will consider *_Multiplicative Weights_* or *_Hedge_*, as well as *_Adaptive Hedge_* as a no-regret algorithm approach to two-player games, as it's regret has been shown in class to be sublinear in T.



### Multiplicative Weights

We will be using the multiplicative weights algorithm often in this project - here, we implement the algorithm as given in Lecture 4. We set our learning rate as follows:


$\eta = \sqrt{\frac{\ln{(K)}}{T}}$


where $K$ is the number of experts we have. Note that we have to know the value of T, our time horizon here in order to get an accurate value for $\eta$.



### Adaptive Hedge

Adaptive hedge is the same as M.W., except we don't know the time horizon T, so we adapt our learning rate as we play. For the purposes of this project, we will follow the procedure outlined in Lecture 12. We start with a learning rate $\eta = 2$, and scale it down by $\phi=2$ at every segment. We play each segment until $\triangle{T}$, the cumulative mixability gap, surpasses our budget $\frac{ln(K)}{\eta}$, then scale down $\eta$, where $K$ is the number of experts we have. This allows us to adaptively learn the best learning rate for our models. We calculate the gap by:



$\triangle{T} = \sum_{t=1}^{T}\delta_t(\eta_t)$, where

$\delta_t(\eta_t) = <w_t, l_t> - (\frac{-1}{\eta_t}\ln{<w_t,e^{-\eta_t l_t}>})$



### Nash and Correlated Equilibrium

A two player game is said to be in Nash Equilibrium when both players have strategies such that if given the strategy of the opponent, no player can decrease their loss any further. This can be found easily for zero-sum games.



Correlated Equilibrium is similar, but not as strict. This is where both players make their strategies given some outside signal, such that they don't know the move of their opponent, but minimize their loss regardless based on this signal. This can be found easily for non-zero-sum games, where Nash Equilibrium would be computationally difficult to find.





## Games - Equilibrium



### Simple 2D Array

Here, we model a game with a 3d reward array $r$. Each player $i$ makes a move $m_i$, and the game is over. The reward for player $i$ is $r[m_0][m_1][i]$.



The first game we will study is the famous non-zero-sum game: the prisonner's dilemna. We define our array $r$ as:



| | Confess | Cooperate |

| -------- | -------- | -------- |

|****Confess**** | -5, -5 | 0, -10 |

|****Cooperate****  |-10, 0 | -1, -1 |



In this representation, player 1 is the row player, and player 2 is the column player.



With weight 0 corresponding to confess and weight 1 corresponding to cooperate, we ran the M.W. algorithm over 50 trials with a time horizon $T=10$. The average weights at each time step is graphed here:



![](https://i.imgur.com/fWcV31L.png)





As we can see, both players quickly converged, and foolishly chose to both confess. We can see that they both converged to a Nash Equilibrium - since both players know that the other will always confess, they have no choice but to confess themselves, as cooperating will net an even higher loss. However, we had a mirrorred reward matrix. What happens if we randomly generate a reward matrix?



| | Action 1 | Action 2 |

| -------- | -------- | -------- |

|****Action 1**** | -0.76, 0.96 | 2.45, 0.47 |

|****Action 2****  |-0.26, -0.57 | 0.24, 3.22 |



![](https://i.imgur.com/bBoiQKH.png)





Here, we set each value of r by sampling from the normal distribution. Again, we see quick convergence of both players' strategies. We ran this over multiple random reward matrices, and in every case, we saw fast convergence. Every case, we also saw a mirroring of weights. We see here that we have again converged to some sort of equilibrium. The row player knows that choosing action 1 will be bad if player 2 always chooses action 1 as well (which will minimize their loss), but player 2 knows that if they were to only choose action 1, player 1 would choose action 1 as well to mitigate loss. Hence, we have some equilibrium where player 2 is willing to sacrifice minimizing loss to keep player 1 consistently choosing action 1. Here, we can see that the probabilty distribution is not absolutely in one direction, yet is still mirrored.



Now, let's try a zero-sum game - we again set the values of r randomly from the normal distribution. However, the reward for player 2 will be exactly the negative of the reward of player 1.



| | Action 1 | Action 2 |

| -------- | -------- | -------- |

|****Action 1**** | 0.65, -0.65 | 1.33, -1.33 |

|****Action 2****  |0.8, -0.8 | -0.41, 0.41 |



![](https://i.imgur.com/0Y6NTzs.png)





Again, we see speedy convergence. It seems that as expected, the m.w. algorithm converges quickly in two player games when playing against an opposing m.w. algorithm. Now, let's move on to a more complex game.



### Blotto Game

The Blotto game is a zero-sum game that revolves around two players, each controlling 100 soldiers. There are ten battlefields, and these players must distribute their soldiers among the ten battlefields. A battlefield is won by the player who sent more soldiers, and the player with the greatest number of won battlefields wins.



We will model our algorithms to follow these rules, but with each expert pertaining to a random distribution across the ten battlefields, and the experts for both players being the same. The loss for each player will be the number of battlefields they lost by (which will be negative if they won). The plot of our loss and weights over time:



![](https://i.imgur.com/hf1uYL6.png)



![](https://i.imgur.com/4Cm9tJI.gif)





As we can see, it appears that the two players never converge on strategy (We ran this for up to 100,000 iterations, and the weights still did not converge)! Since they have the same experts, their play can be mirrored - as player 1 puts more weights towards experts that help them beat player 2's strategy, player 2 would counter. We can see this in the moves the two players make over time, as they change around to suit each other. If we were to plot the graph for t=100000, we would see a lot of mirrored oscilation around losses.



However, we ran this with 50 experts, which is a ton of dimensions over just having 2 experts. What if we only had 10 experts?



![](https://i.imgur.com/O1Ziz7s.png)



![](https://i.imgur.com/601IwEv.gif)



It seems that we actually do converge with 10 experts! Perhaps, 50 experts do eventually converge, but we can not run enough iterations for it to do so. As we can see, the losses converge as well - both players reach Nash Equilibrium, and their losses stagnate as they both use the exact same experts. At this point, both players incur on average 0 loss.



Now, what if we have different experts for both players? Let's see:



![](https://i.imgur.com/BNF0MJA.png)



![](https://i.imgur.com/OmM7aPs.gif)



Interesting! It looks like with a different distribution of experts, our losses still converge in direction! However, our weights do not converge - as soon as one strategy dominates, the other player comes up with an expert to counter this strategy, and so on. In this case, we have not reached Nash or any equivalent equilibrium, although this can be seen as some sort of oscillating equilibrium. The reason our losses converge in direction is most likely due to the fact that one set of randomly generated experts just happens to be better than the other set - here, no matter how hard player 2 tries, their strategies are just not good enough against player 1. Whereas before, with the same set of experts, this oscillation in strategy countering caused the losses of both players to oscillate around zero. Although, it's possible that these strategies eventually converge at one point - we have ran up to 50,000 iterations without signs of convergence, and it may be possible that convergence happens with more iterations.





## Games - Adaptive Hedge



Now, let's try running AdaHedege on the Blotto game. For these experiments, we will run with 10 experts per model. First, let's see how AdaHedge does against itself, with the same experts:



![](https://i.imgur.com/GftDWRR.png)

![](https://i.imgur.com/VvOlO1L.png)

![](https://i.imgur.com/vz1A5f1.gif)



It seems that AdaHedge converges rather quickly against itself to find a Nash equilibrium! We can see that the learning rate only updates twice, and we only have three segments - we can also see that despite having polar opposite losses, both models' learning rates shifted at the same time, which is very interesting. This implies that even though the first player was consistently beating the second player, their other experts (that they didn't pick) were incurring loss on a level similar to the second player, resulting in both starting new segments at around the same time.



Now, let's see how AdaHedge does against normal M.W. with the same 10 experts:



![](https://i.imgur.com/OyXbsQ3.png)

![](https://i.imgur.com/8pX5Xix.png)

![](https://i.imgur.com/pptJ0xc.gif)



We see that once again, we converge to a Nash equilibrium after some time. However, it is interesting to see that AdaHedge creates twice as many segments when playing against M.W. than when it was playing against itself - comparing against the plot of losses, we see that AdaHedge actually only creates these segments while it's cumulative loss is lower than M.W., and afterwards, sticks to the same learning rate. This can be due to the fact that M.W. was initialized with a good learning rate as we already knew the time horizon T, while AdaHedge had to adaptively tune to a good learning rate. As a result, M.W. converged to the best strategy much faster than AdaHedge, resulting in higher losses and more learning rate changes for AdaHedge than if it was playing against itself.



## Conclusion



From our results, we can see generally in zero-sum games, playing no-regret algorithms such as Multiplicative Weights and Adaptive Hedge against each other will ultimately converge in Nash Equilibrium, with the weights and losses converging. However, in some cases, such as the Blotto game with 50 experts, this convergence can take a very long time, with time to convergence growing seemingly exponentially with the number of experts to choose from.



We saw that when AdaHedge plays against itself, despite which player comes up with a better strategy, they create new segments at around the same time and of similar length. When AdaHedge plays against the Multiplicative Weights algorithm, it creates a lot more segments and performs more poorly. This is expected, as M.W. has access to the true value of T, while AdaHedge can only approximate it by adapting its learning rate. In both cases, both players eventually converge to a Nash Equilibrium, which is expected, as both strategies are no-regret strategies, to an order that is sublinear in T.



If I could have more time for this project, I would definitely go and explore how batching online learning would change things, for our third objective. Right now, our models update their weights after each iteration, or each game played - what if we batched updates, such that we would play for n games, then update our weights *_after_* all n games have finished? What if we applied some sort of batch normalization to these batches, and how would these impact our models' abilities to adapt to covariance shift? I would test this by having a covariance shift happen at T/2 iterations, and test the speed of convergence with and without batch normalization.



In the future, I would also like to incorporate more games, both zero-sum and non-zero sum - perhaps a guessing game between 0 to 100, where the true number follows some underlying distribution. It would be interesting to see how fast AdaHedge and Multiplicative Weights would converge to this true underlying distribution as they play.
