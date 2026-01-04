import numpy as np
import matplotlib.pyplot as plt
import math
import random 


## PROBLEM 2 : BANDITS
## In this section, we have given you a template for coding each of the 
## exploration algorithms: epsilon-greedy, optimistic initialization, UCB exploration, 
## and Boltzmann Exploration 

## You will be implementing these algorithms as described in the “10-armed Testbed” in Sutton+Barto Section 2.3
## Please refer to the textbook or office hours if you have any confusion.

## note: you are free to change the template as you like, do not think of this as a strict guideline
## as long as the algorithm is implemented correctly and the reward plots are correct, you will receive full credit

# This is the optional wrapper for exploration algorithm we have provided to get you started
# this returns the expected rewards after running an exploration algorithm in the K-Armed Bandits problem
# we have already specified a number of parameters specific to the 10-armed testbed for guidance
# iterations is the number of times you run your algorithm

# WRAPPER FUNCTION
def explorationAlgorithm(explorationAlgorithm, param, iters):
    cumulativeRewards = np.zeros((iters,1000))
    for i in range(iters):
        # number of time steps
        t = 1000
        # number of arms, 10 in this instance
        k = 10
        # real reward distribution across K arms
        rewards = np.random.normal(1,1,k)
        # counts for each arm
        n = np.zeros(k)
        # extract expected rewards by running specified exploration algorithm with the parameters above
        # param is the different, specific parameter for each exploration algorithm
        # this would be epsilon for epsilon greedy, initial values for optimistic intialization, c for UCB, and temperature for Boltmann 
        currentRewards = explorationAlgorithm(param, t, k, rewards, n)
        cumulativeRewards[i] = currentRewards
    # TO DO: CALCULATE AVERAGE REWARDS ACROSS EACH ITERATION TO PRODUCE EXPECTED REWARDS
    expectedRewards = np.mean(cumulativeRewards, axis=0)
    return expectedRewards

# EPSILON GREEDY TEMPLATE
def epsilonGreedy(epsilon, steps, k, realRewards, n):
    # TO DO: initialize structure to hold expected rewards per step
    rs = np.zeros(steps)
    # TO DO: initialize an initial q value for each arm
    Qs = np.zeros(k)
    # TO DO: implement the epsilon-greedy algorithm over all steps and return the expected rewards across all steps
    for i in range(steps):
        prob = random.random()
        if prob >= epsilon:
            a = np.argmax(Qs)
        else:
            a = random.randint(0, k-1)
        r = np.random.normal(realRewards[a],1)
        n[a] += 1
        Qs[a] += 1/n[a] * (r-Qs[a])
        
        rs[i] = (1-epsilon) * realRewards[np.argmax(Qs)] + np.sum(epsilon * 1/k * realRewards)
    return rs

# OPTIMISTIC INTIALIZATION TEMPLATE
def optimisticInitialization(value, steps, k, realRewards, n):
    # TO DO: initialize structure to hold expected rewards per step
    rs = np.zeros(steps)
    # TO DO: initialize optimistic initial q values per arm specified by parameter
    Qs = np.ones(k) * value
    # TO DO: implement the optimistic initializaiton algorithm over all steps and return the expected rewards across all steps
    for i in range(steps):
        a = np.argmax(Qs)
        r = np.random.normal(realRewards[a],1)
        n[a] += 1
        Qs[a] += 1/n[a] * (r - Qs[a])

        rs[i] = realRewards[np.argmax(Qs)]
    return rs
# UCB EXPLORATION TEMPLATE
def ucbExploration(c, steps, k, realRewards, n):
    # TO DO: initialize structure to hold expected rewards per step
    rs = np.zeros(steps)
    # TO DO: initialize q values per arm 
    Qs = np.zeros(k)
    # TO DO: implement the UCB exploration algorithm over all steps and return the expected rewards across all steps
    for i in range(steps):
        if i < k:
            a = i
        else:
            a = np.argmax(Qs + c * np.sqrt(np.log(steps)/n))
        r = np.random.normal(realRewards[a],1)
        n[a] += 1     
        Qs[a] += 1/n[a] * (r - Qs[a])
        if i < k:
            rs[i] = realRewards[i]
        else:
            rs[i] = realRewards[np.argmax(Qs + c * np.sqrt(np.log(steps)/n))]
    return rs

# BOLTZMANN EXPLORATION TEMPLATE
def boltzmannE(temperature, steps, k, realRewards, n):
    # TO DO: initialize structure to hold expected rewards per step
    rs = np.zeros(steps)
    # TO DO: initialize q values per arm 
    Qs = np.ones(k)
    # TO DO: initialize probability values for each arm
    probs = np.ones(k) * 1/k
    # TO DO: implement the Boltzmann Exploration algorithm over all steps and return the expected rewards across all steps
    for i in range(steps):
        rand = random.uniform(0,1)
        prob_sum = 0
        for j, prob in enumerate(probs):
            prob_sum += prob
            if rand <= prob_sum:
                a = j
                break
        r = np.random.normal(realRewards[a],1)
        n[a] += 1
        Qs[a] += 1/n[a] * (r - Qs[a])

        denom = 0
        for l in range(k):
            val = math.exp(temperature*Qs[l])
            probs[l] = val
            denom += val
        probs = probs/denom
        rs[i] = np.sum(np.multiply(realRewards,probs))
    return rs

# PLOT TEMPLATE
def plotExplorations(paramList, Algorithm, exploreName, paramName):
    # TO DO: for each parameter in the param list, plot the returns from the exploration Algorithm from each param on the same plot
    x = np.arange(1,1001)
    # calculate your Ys (expected rewards) per each parameter value
    # plot all the Ys on the same plot
    # include correct labels on your plot!
    for param in paramList:
        y = explorationAlgorithm(Algorithm,param, 100)
        plt.plot(x,y, label= "%s = %.03f" % (paramName, param))
    plt.legend()
    plt.title(exploreName)
    plt.show()

if __name__ == "__main__":
    plotExplorations([0,.001,.01,.1,1], epsilonGreedy, "Epsilon Greedy Exploration Aglorithm", "epsilon")
    plotExplorations([0,1,2,5,10], optimisticInitialization, "Optimistic Initialization Exploration Algorithm", "inital value")
    plotExplorations([0,1,2,5], ucbExploration, "UCB Exploration Algorithm", "c")
    plotExplorations([1,3,10,30,100],boltzmannE, "Boltzmann Exploration Algorithm", "temperature")
    x = np.arange(1,1001)
    y = explorationAlgorithm(epsilonGreedy, .1, 100)
    plt.plot(x,y, label="Epsilon greedy: epsilon = 0.1")
    y = explorationAlgorithm(optimisticInitialization, 5, 100)
    plt.plot(x,y, label="Optimistic Initialization: initalization = 5")
    y = explorationAlgorithm(ucbExploration, 2, 100)
    plt.plot(x,y,label="UCB: c = 2")
    plt.legend()
    plt.title("Best Performance Comparision for Each Exploration Strategy")
    plt.show()