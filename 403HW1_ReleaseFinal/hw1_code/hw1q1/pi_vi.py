# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gym
import lake_envs as lake_env
from queue import PriorityQueue


def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)

    print(str_policy)


def value_function_to_policy(env, gamma, value_function):
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """
    # Hint: You might want to first calculate Q value,
    #       and then take the argmax.
    policy = np.zeros(env.nS, dtype='int')
    for s in range(env.nS):
      max_action_val = -1
      for a in range(env.nA):
        cur_action_val = 0
        nexts = env.P[s][a]
        for prob, nextstate, reward, is_terminal in nexts:
          if is_terminal:
            cur_action_val += prob * reward
          else:
            cur_action_val += prob * (reward + gamma * value_function[nextstate])
        if cur_action_val > max_action_val:
          policy[s] = a
          max_action_val = cur_action_val
    return policy


def evaluate_policy_sync(env, value_func, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.

    Evaluates the value of a given policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    value_func: np.array
      The current value functione estimate
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    value_func_k = value_func
    value_func_k_1 = np.zeros_like(value_func)
    counter = 0
    while (1):
      counter +=1
      delta = 0
      for s in range(env.nS):
        v = value_func_k[s]
        nexts = env.P[s][policy[s]]
        V_s = 0
        for prob, nextstate, reward, is_terminal in nexts:
          if is_terminal:
            V_s += prob * reward
          else:
            V_s += prob * (reward + gamma * value_func_k[nextstate])
        value_func_k_1[s] = V_s
        delta = max(delta, abs(v - value_func_k_1[s]))
      value_func_k = value_func_k_1
      if counter == max_iterations or delta < tol: break
    return value_func_k, counter


def evaluate_policy_async_ordered(env, value_func, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.

    Evaluates the value of a given policy by asynchronous DP.  Updates states in
    their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    value_func: np.array
      The current value functione estimate
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    counter = 0
    while (1):
      counter +=1
      delta = 0
      for s in range(env.nS):
        v = value_func[s]
        nexts = env.P[s][policy[s]]
        V_s = 0
        for prob, nextstate, reward, is_terminal in nexts:
          if is_terminal:
            V_s += prob * reward
          else:
            V_s += prob * (reward + gamma * value_func[nextstate])
        value_func[s] = V_s
        delta = max(delta, abs(v - value_func[s]))
      if counter == max_iterations or delta < tol: break
    return value_func, counter


def evaluate_policy_async_randperm(env, value_func, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.

    Evaluates the value of a policy.  Updates states by randomly sampling index
    order permutations.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    value_func: np.array
      The current value functione estimate
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    counter = 0
    rng = np.random.default_rng()
    random_state = np.arange(env.nS, dtype="int")
    rng.shuffle(random_state)
    while (1):
      counter +=1
      delta = 0
      for s in random_state:
        v = value_func[s]
        nexts = env.P[s][policy[s]]
        V_s = 0
        for prob, nextstate, reward, is_terminal in nexts:
          if is_terminal:
            V_s += prob * reward
          else:
            V_s += prob * (reward + gamma * value_func[nextstate])
        value_func[s] = V_s
        delta = max(delta, abs(v - value_func[s]))
      if counter == max_iterations or delta < tol: break
    return value_func, counter

def improve_policy(env, gamma, value_func, policy):
    """Performs policy improvement.

    Given a policy and value function, improves the policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """
    stable = True
    for s in range(env.nS):
      old_action = policy[s]
      max_action_val = -1
      for a in range(env.nA):
        cur_action_val =0
        nexts = env.P[s][a]
        for prob, nextstate, reward, is_terminal in nexts:
          if is_terminal:
            cur_action_val += prob * reward
          else:
            cur_action_val += prob * (reward + gamma * value_func[nextstate])
        if cur_action_val > max_action_val:
          policy[s] = a
          max_action_val = cur_action_val

      if old_action != policy[s]:
        stable = False
    return stable, policy


def policy_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 85 of the Sutton & Barto Second Edition book.

    You should use the improve_policy() and evaluate_policy_sync() methods to
    implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.random.randint(0, env.nA, env.nS, dtype='int')
    value_func = np.zeros(env.nS)
    policy_counter = 0
    value_counter = 0

    while(1):
      value_func , v_counter = evaluate_policy_sync(env,value_func,gamma,policy,max_iterations,tol)
      value_counter += v_counter

      stable, policy = improve_policy(env,gamma,value_func,policy)
      policy_counter += 1
      if stable: break
    return policy, value_func, policy_counter, value_counter


def policy_iteration_async_ordered(env, gamma, max_iterations=int(1e3),
                                   tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_ordered methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.random.randint(0, env.nA, env.nS, dtype='int')
    value_func = np.zeros(env.nS)
    policy_counter = 0
    value_counter = 0

    while(1):
      value_func , v_counter = evaluate_policy_async_ordered(env, value_func, gamma,policy,max_iterations,tol)
      value_counter += v_counter

      stable, policy = improve_policy(env,gamma,value_func,policy)
      policy_counter += 1
      if stable: break
    return policy, value_func, policy_counter, value_counter


def policy_iteration_async_randperm(env, gamma, max_iterations=int(1e3),
                                    tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_randperm methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.random.randint(0, env.nA, env.nS, dtype='int')
    value_func = np.zeros(env.nS)
    policy_counter = 0
    value_counter = 0

    while(1):
      value_func , v_counter = evaluate_policy_async_randperm(env, value_func, gamma,policy,max_iterations,tol)
      value_counter += v_counter

      stable, policy = improve_policy(env,gamma,value_func,policy)
      policy_counter += 1
      if stable: break
    return policy, value_func, policy_counter, value_counter

def value_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func_k = np.zeros(env.nS)  # initialize value function
    value_func_k_1 = np.zeros(env.nS)
    counter = 0
    while(1):
      counter += 1
      delta = 0
      for s in range(env.nS):
        v = value_func_k[s]
        max_action_val = 0
        for a in range(env.nA):
          cur_action_val = 0
          nexts = env.P[s][a]
          for prob, nextstate, reward, is_terminal in nexts:
            if is_terminal:
              cur_action_val += prob * reward
            else:
              cur_action_val += prob * (reward + gamma * value_func_k[nextstate])
          if cur_action_val > max_action_val:
            value_func_k_1[s] = cur_action_val
            max_action_val = cur_action_val
        delta = max(delta, abs(v - value_func_k_1[s]))
      value_func_k = value_func_k_1
      if delta < tol or counter == max_iterations: break
    return value_func_k, counter


def value_iteration_async_ordered(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states in their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)  # initialize value function
    counter = 0
    while(1):
      counter += 1
      delta = 0
      for s in range(env.nS):
        v = value_func[s]
        max_action_val = 0
        for a in range(env.nA):
          cur_action_val = 0
          nexts = env.P[s][a]
          for prob, nextstate, reward, is_terminal in nexts:
            if is_terminal:
              cur_action_val += prob * reward
            else:
              cur_action_val += prob * (reward + gamma * value_func[nextstate])
          if cur_action_val > max_action_val:
            value_func[s] = cur_action_val
            max_action_val = cur_action_val
        delta = max(delta, abs(v - value_func[s]))
      if delta < tol or counter == max_iterations: break
    return value_func, counter


def value_iteration_async_randperm(env, gamma, max_iterations=int(1e3),
                                   tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by randomly sampling index order permutations.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)  # initialize value function
    rng = np.random.default_rng()
    random_state = np.arange(env.nS, dtype="int")
    rng.shuffle(random_state)
    counter = 0
    while(1):
      counter += 1
      delta = 0
      for s in random_state:
        v = value_func[s]
        max_action_val = 0
        for a in range(env.nA):
          cur_action_val = 0
          nexts = env.P[s][a]
          for prob, nextstate, reward, is_terminal in nexts:
            if is_terminal:
              cur_action_val += prob * reward
            else:
              cur_action_val += prob * (reward + gamma * value_func[nextstate])
          if cur_action_val > max_action_val:
            value_func[s] = cur_action_val
            max_action_val = cur_action_val
        delta = max(delta, abs(v - value_func[s]))
      if delta < tol or counter == max_iterations: break
    return value_func, counter


def value_iteration_async_custom(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by student-defined heuristic.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)  # initialize value function
    PQueue = PriorityQueue()
    state_by_dist = []
    if (env.nS == 16):
      goal_row = 1
      goal_collumn = 1
      for s in range(env.nS):
        row = int(s / 4)
        collumn = s % 4
        PQueue.put((abs(row-goal_row)+abs(collumn-goal_collumn),s))
    else:
      goal_row = 1
      goal_collumn = 7
      for s in range(env.nS):
        row = int(s / 8)
        collumn = s % 8
        PQueue.put((abs(row-goal_row)+abs(collumn-goal_collumn),s))
    while not PQueue.empty():
      _, s = PQueue.get()
      state_by_dist.append(s)
    counter = 0
    while(1):
      counter += 1
      delta = 0
      for s in state_by_dist:
        v = value_func[s]
        max_action_val = 0
        for a in range(env.nA):
          cur_action_val = 0
          nexts = env.P[s][a]
          for prob, nextstate, reward, is_terminal in nexts:
            if is_terminal:
              cur_action_val += prob * reward
            else:
              cur_action_val += prob * (reward + gamma * value_func[nextstate])
          if cur_action_val > max_action_val:
            value_func[s] = cur_action_val
            max_action_val = cur_action_val
        delta = max(delta, abs(v - value_func[s]))
      if delta < tol or counter == max_iterations: break
    return value_func, counter


######################
#  Optional Helpers  #
######################

# Here we provide some helper functions simply for your convinience.
# You DON'T necessarily need them, especially "env_wrapper" if
# you want to deal with it in your different ways.

# Feel FREE to change/delete these helper functions.

def display_policy_letters(env, policy):
    """Displays a policy as letters, as required by problem 1.2 & 1.3

    Parameters
    ----------
    env: gym.core.Environment
    policy: np.ndarray, with shape (env.nS)
    """
    policy_letters = []
    for l in policy:
        policy_letters.append(lake_env.action_names[l][0])

    policy_letters = np.array(policy_letters).reshape(env.nrow, env.ncol)


    for row in range(env.nrow):
        print(''.join(policy_letters[row, :]))


def env_wrapper(env_name):
    """Create a convinent wrapper for the loaded environment

    Parameters
    ----------
    env: gym.core.Environment

    Usage e.g.:
    ----------
        envd4 = env_load('Deterministic-4x4-FrozenLake-v0')
        envd8 = env_load('Deterministic-8x8-FrozenLake-v0')
    """
    env = gym.make(env_name)

    # T : the transition probability from s to s’ via action a
    # R : the reward you get when moving from s to s' via action a
    env.T = np.zeros((env.nS, env.nA, env.nS))
    env.R = np.zeros((env.nS, env.nA, env.nS))

    for state in range(env.nS):
      for action in range(env.nA):
        for prob, nextstate, reward, is_terminal in env.P[state][action]:
            env.T[state, action, nextstate] = prob
            env.R[state, action, nextstate] = reward
    return env


def value_func_heatmap(env, value_func):
    """Visualize a policy as a heatmap, as required by problem 1.2 & 1.3

    Note that you might need:
        import matplotlib.pyplot as plt
        import seaborn as sns

    Parameters
    ----------
    env: gym.core.Environment
    value_func: np.ndarray, with shape (env.nS)
    """
    fig, ax = plt.subplots(figsize=(7,6))
    sns.heatmap(np.reshape(value_func, [env.nrow, env.ncol]),
                annot=False, linewidths=.5, cmap="GnBu_r", ax=ax,
                yticklabels = np.arange(1, env.nrow+1)[::-1],
                xticklabels = np.arange(1, env.nrow+1))
    plt.show()
    # Other choices of cmap: YlGnBu
    # More: https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
    return None


if __name__ == "__main__":
    envs = ['Deterministic-4x4-FrozenLake-v0', 'Deterministic-8x8-FrozenLake-v0']
    # Define num_trials, gamma and whatever variables you need below.
    for name in envs:
      env = gym.make(name)
      policy, value_func, policy_counter, value_counter = policy_iteration_sync(env, 0.9, tol=1e-3)
      print('%s: Synchrous Policy interation: improve steps: %d eval steps: %d' % (name, policy_counter, value_counter))
      display_policy_letters(env, policy)
      print()
      value_func_heatmap(env, value_func)
      value_func, iterations = value_iteration_sync(env, 0.9, tol = 1e-3)
      print("%s: Synchronous Value interation: iterations: %d" % (name, iterations))
      value_func_heatmap(env, value_func)
      display_policy_letters(env, value_function_to_policy(env, 0.9, value_func))
      print()
      if name == 'Deterministic-8x8-FrozenLake-v0':
        _, _, policy_counter, value_counter = policy_iteration_async_ordered(env, 0.9, tol=1e-3)
        print("%s: Asynchronous Ordered Policy iteration: improve steps: %d eval steps: %d" % (name, policy_counter, value_counter))
        print()
        improve_steps = 0
        eval_steps = 0
        for i in range(10):
          _, _, improve_counter, eval_counter = policy_iteration_async_randperm(env, 0.9, tol=1e-3)
          improve_steps += improve_counter
          eval_steps += eval_counter
        improve_steps /= 10
        eval_steps /= 10
        print("%s: Asynchronous Randperm Policy iteration: improves steps: %d eval steps: %s" % (name, improve_steps, eval_steps))
        print()
      _, interations = value_iteration_async_ordered(env, 0.9, tol=1e-3)
      print("%s: Asynchronous Ordered Value iteration: iterations %d" % (name, interations))
      print()
      counter = 0
      for i in range(10):
        _, interations = value_iteration_async_randperm(env, 0.9, tol=1e-3)
        counter += interations
      counter /= 10
      print("%s: Asynchronous Randperm Value iteration: iterations: %d" % (name, counter))
      print()
      _, iterations = value_iteration_async_custom(env, 0.9, tol=1e-3)
      print("%s: Asynchronous Custom Value iteration: iterations: %d" % (name, iterations))
      print()
