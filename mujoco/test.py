import gym
import numpy as np

env = gym.make('FetchReach-v1')

env.reset()
print(env.action_space)
print(env.action_space.sample())
for i in range(10000):
    #render wont work
    #env.render()
    if i % 500 == 0:
        u1 = np.random.uniform(0, 1, 3)
        u2 = np.random.uniform(0, 1, 3)
        u3 = np.random.uniform(0, 1, 3)
        u1 = u1/np.sum(u1)
        u2 = u2/np.sum(u2)
        u3 = u3/np.sum(u3)

    vals = [0, 1, 2]
    i1 = np.random.choice(vals, p=u1)
    i2 = np.random.choice(vals, p=u1)
    i3 = np.random.choice(vals, p=u1)

    a1 = [0, 0.2, -0.2][i1]
    a2 = [0, 0.2, -0.2][i2]
    a3 = [0, 0.2, -0.2][i3]

    ans = env.step([a1, a2, a3, 0])
    obs = ans[0]["observation"]
    print(ans)
    print(obs)

    if obs[0] <= 1.00 or obs[0] >= 1.510:
        env.reset()
    elif obs[1] <= 0.38 and obs[1] >= 1.1:
        env.reset()
    elif obs[2] >= 0.875:
        env.reset()
    #  print(ans[0]["observation"])


# [1.03-1.5, 0.38-1.1, 0.415.25]
