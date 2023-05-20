import numpy as np
import gym
import time
import matplotlib.pyplot as plt
#<---------------------------------------------Q-LEARNING ALGO START --------------------------------------------->
print("Q-Learning Started")
# Define the environment
env = gym.make("Taxi-v3").env

# Initialize the q-table with zero values Q(s,a)
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1  # learning-rate
gamma = 0.7  # discount-factor
epsilon = 0.1  # explore vs exploit
max_episode_length = 50000  # max episode length
N = 10000  # Total episodes

rng = np.random.default_rng()  # Random generator

# Reward list
rewards_arr = []
num_episodes_arr = []

lst = list(range(1, N + 1))
for i in lst:
    rewards = []
    # Reset the environment
    state, info = env.reset()

    done = False

    # Keep Iterating until the terminal state is reached
    lgt = 0
    while not done:
        if lgt > max_episode_length:
            break
        if rng.random() < epsilon:
            action = env.action_space.sample()  # Explore the action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values

        # Apply the action and see what happens
        next_state, reward, done, flag, info = env.step(action)

        current_value = q_table[
            state, action]  # current Q-value for the state/action couple
        next_max = np.max(q_table[next_state])  # next best Q-value

        # Append reward to reward list
        rewards.append(reward)

        # Compute the new Q-value with the Bellman equation
        q_table[state, action] = (1 - alpha) * current_value + alpha * (
            reward + gamma * next_max)

        # Next state becomes current state
        state = next_state
        lgt += 1
    rewards_arr.append(sum(rewards) / len(rewards))
    num_episodes_arr.append(lgt)

print("Q-Table for Q-Learning: ")
print(q_table)
print("Average Reward for Q-Learning Algorithm: ", np.mean(rewards_arr))

Q_REWARD = rewards_arr.copy()
Q_LENGTH = num_episodes_arr.copy()
print("Q-Learning Ended")
#<----------------------------------------------------Q-LEARNING ALGO END----------------------------------------------------->

#<----------------------------------------------------SARSA ALGO START----------------------------------------------------->
print("SARSA Started")
# Define the environment
env = gym.make("Taxi-v3").env

# Initialize the q-table with zero values
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1  # learning-rate
gamma = 0.7  # discount-factor
epsilon = 0.1  # explore vs exploit
max_episode_length = 50000

# Random generator
rng = np.random.default_rng()


# Action Selector Function
def choose_action(state):
    if rng.random() < epsilon:
        action = env.action_space.sample()  # Explore the action space
    else:
        action = np.argmax(q_table[state])  # Exploit learned values
        # print(int(action))
    return action


# Rewards list
rewards_arr = []
num_episodes_arr = []
N = 10000
lst = list(range(1, N + 1))
for i in lst:
    # Reset the environment
    state, info = env.reset()
    rewards = []
    done = False
    lgt = 0
    # Loop as long as the game is not over, i.e. done is not True
    while not done:

        # Select action
        action = choose_action(state)

        # Apply the action and see what happens
        next_state, reward, done, flag, info = env.step(action)

        current_value = q_table[
            state, action]  # current Q-value for the state/action couple

        next_action = choose_action(next_state)  # next action
        next_val = q_table[next_state, next_action]  # next Q-value

        # Append reward to reward list
        rewards.append(reward)

        # Compute the new Q-value with the Bellman equation
        q_table[state, action] = (1 - alpha) * current_value + alpha * (
            reward + gamma * next_val)

        # Next state and action becomes current state and action
        state = next_state
        action = next_action
        if lgt > max_episode_length:
            break
        lgt += 1
    rewards_arr.append(sum(rewards) / len(rewards))
    num_episodes_arr.append(lgt)

print("Q-Table for SARSA: ")
print(q_table)
print("Average Reward for SARSA: ", np.mean(rewards_arr))

SARSA_REWARD = rewards_arr.copy()
SARSA_LENGTH = num_episodes_arr.copy()
print("SARSA ended")
#<----------------------------------------------------SARSA ALGO END----------------------------------------------------->

#<----------------------------------------------------MC Prediction ALGO START----------------------------------------------------->
print("MC Started")
# Define the environment
env = gym.make("Taxi-v3").env

# Initialize V(s), pi(s), Returns(s)
v_table = np.zeros([env.observation_space.n])
policy = dict()
returns = dict()
for i in range(0, 500):
    policy[i] = env.action_space.sample(env.action_mask(i))
    returns[i] = []

# Hyperparameters
gamma = 0.7  # discount-factor
max_episode_length = 50000

# Reward list
rewards_arr = []
num_episodes_arr = []
N = 1000
lst = list(range(1, N + 1))
for k in lst:
    #  as the game is not over, i.e. done is not True
    lgt = 0
    done = False
    # Reset the environment
    state, info = env.reset()
    episode = [state]
    episodeReward = [0]
    rnd = False
    while not done:  # Episode Generation
        if k == 1 or rnd:
            action = env.action_space.sample(env.action_mask(state))
        else:
            action = policy[state]
        next_state, reward, done, flag, info = env.step(action)
        # print(next_state)
        if done:
            break
        episode.append(next_state)
        episodeReward.append(reward)
        state = next_state
        if state == next_state:
            rnd = True
        else:
            rnd = False
        if lgt > max_episode_length:
            break
        lgt += 1
        # print(lgt)
    # print(lgt)
    num_episodes_arr.append(lgt)

    G = 0
    for i in range(len(episode)):
        # Calculating G
        G = episodeReward[i] + gamma * G
        returns[episode[i]].append(G)
        tmp = v_table[episode[i]]
        # Updating V(s)
        v_table[episode[i]] = sum(returns[episode[i]]) / len(
            returns[episode[i]])
    # Policy Improvement
    for z in range(0, 500):
        a = policy[z]
        max_value = -float('inf')
        max_action = None
        for action in range(6):
            next_state, reward, done, flag, info = env.step(action)
            tmp = reward  #+ gamma * v_table[next_state]
            if tmp > max_value:
                max_value = tmp
                max_action = action
        policy[z] = max_action
    rewards_arr.append(sum(episodeReward) / len(episodeReward))

print("V(s): ")
print(v_table)
print("Average Reward for MC Every-Visit: ", np.mean(rewards_arr))
MC_REWARD = rewards_arr.copy()
MC_LENGTH = num_episodes_arr.copy()
print("MC Ended")
#<----------------------------------------------------MC Prediction ALGO END----------------------------------------------------->

#<---------------------------------------------------------Plotting--------------------------------------------------------------->
plt.figure()
plt.plot(lst[:N], Q_REWARD[:N])
plt.plot(lst[:N], SARSA_REWARD[:N])
plt.plot(lst[:N], MC_REWARD[:N])
plt.legend(["Q-Learning", "SARSA", "MC Every Visit"])
plt.title("Reinforcement Learning on Taxi Environment")
plt.xlabel("Episode Number")
plt.ylabel("Average Reward for an episode")

plt.figure()
plt.title("Monte-Carlo on Taxi Environment")
plt.plot(lst[:N], MC_REWARD[:N])
plt.xlabel("Episode Number")
plt.ylabel("Average Reward for an episode")

plt.figure()
plt.title("Q-Learning on Taxi Environment")
plt.plot(lst[:N], Q_REWARD[:N])
plt.xlabel("Episode Number")
plt.ylabel("Average Reward for an episode")

plt.figure()
plt.title("SARSA on Taxi Environment")
plt.plot(lst[:N], SARSA_REWARD[:N])
plt.xlabel("Episode Number")
plt.ylabel("Average Reward for an episode")

plt.figure()
plt.plot(lst[:N], Q_LENGTH[:N])
plt.plot(lst[:N], SARSA_LENGTH[:N])
plt.plot(lst[:N], MC_LENGTH[:N])
plt.legend(["Q-Learning", "SARSA", "MC Every Visit"])
plt.title("Reinforcement Learning on Taxi Environment")
plt.xlabel("Episode Number")
plt.ylabel("Number of Steps (epoch)")

plt.figure()
plt.title("Monte-Carlo on Taxi Environment")
plt.plot(lst[:N], MC_LENGTH[:N])
plt.xlabel("Episode Number")
plt.ylabel("Number of Steps (epoch)")

plt.figure()
plt.title("Q-Learning on Taxi Environment")
plt.plot(lst[:N], Q_LENGTH[:N])
plt.xlabel("Episode Number")
plt.ylabel("Number of Steps (epoch)")

plt.figure()
plt.title("SARSA on Taxi Environment")
plt.plot(lst[:N], SARSA_LENGTH[:N])
plt.xlabel("Episode Number")
plt.ylabel("Number of Steps (epoch)")

plt.show()