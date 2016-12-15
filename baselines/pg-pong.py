""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import cPickle as pickle
import gym
from PIL import Image

res_dir = './results/'

# hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = False  # resume from previous checkpoint?
render = False

# model initialization
D = 80 * 80  # input dimensionality: 80x80 grid  After pre-processing, crop and downsampling the 210x160x3 to 80x80 image.
if resume:
    model = pickle.load(open(res_dir + 'save.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization done.
    model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = {k: np.zeros_like(v) for k, v in model.iteritems()}  # update buffers that add up gradients over a batch
# iterate key, values in dictionary model, grad_buffer = {'W1': all zeros matrix that has same size as model['W1'],
# 'W2: all zeros matrix that has same size as model['W2']'}
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.iteritems()}  # rmsprop memory


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop the original image to 160x160x3
    I = I[::2, ::2, 0]  # downsample by factor of 2    from 160x160x3 to 80x80, get the first slice of tensor
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1

    return I.astype(np.float).ravel()


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)  # vector Nx1 , N is the number of states in this episode.
    running_add = 0
    for t in reversed(xrange(0, r.size)):  # suppose r.size =3, then iterate t in order 2,1,0
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add    # At current time step t, the cumulative rewards from t to the end of episode.
    return discounted_r


def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h  # return probability of taking action 2, and hidden state


def policy_backward(eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp).ravel()  # ravel return an array. eph.T = transpose of eph. np.dot(A,B) = A*B
    dh = np.outer(epdlogp, model['W2'])  # np.outer(a,b) outer product of a: Nx1 and b: Mx1 => A = np.outer(a,b) NxM Aij = ai*bj
    dh[eph <= 0] = 0  # backpro prelu.
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}


env = gym.make("Pong-v0")
# env.monitor.start('./results/Pong-experiment-1') # record results for uploading

observation = env.reset()
observation, reward, done, info = env.step(2)

prev_x = None  # used in computing the difference frame
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
while True:
    if render: env.render()

    # img = Image.fromarray(observation, 'RGB')
    # img.show()


    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)

    x = cur_x - prev_x if prev_x is not None else np.zeros(D) # if prev_x is not none, return cur_x - prev_x  else return cur_x
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3  # roll the dice! sample action  uniformly draw value from [0, 1)

    # record various intermediates (needed later for backprop)
    xs.append(x)  # observation  xs = [[x1],[x2],[x3]...] for one episode. x1 = 1x6400
    hs.append(h)  # hidden state hs = [h1,h2,h3,...] h1 = 1x200 array, hs = Nx200 matrix
    y = 1 if action == 2 else 0  # a "fake label": if the correct action is 2. then the true prob should be 1, the label should be 1.
    dlogps.append(
        y - aprob)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
    #  gradients for one episode. [e1, e2, e3,...], e1 = y - aprob

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)
    # record reward from t0 to the end of episode, all the reward for one episode.

    if done:  # an episode finished
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs) # Stack arrays in sequence vertically (row wise).
        eph = np.vstack(hs) # eph = array([[1, 2, 3, ...200], [1, 2, 3, 4,...200]]), where h1 = [1,,2,3,...200]
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, dlogps, drs = [], [], [], []  # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)  # N states for one episode: Nx1 discount_reward vector
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
        # f(x)(y - aprob), f is the reward function (discounted_epr).

        grad = policy_backward(eph, epdlogp)  # calculate gradient of W1 and W2 for one epsiode.
        for k in model: grad_buffer[k] += grad[k]  # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for k, v in model.iteritems():
                g = grad_buffer[k]  # gradient {'W1': dW1, 'W2': dW2}
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2   # root mean square gradient
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

        # boring book-keeping  1. episode_reward: average reward from beginning until now.
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
        if episode_number % 100 == 0:
            pickle.dump(model, open(res_dir + 'save_new.p', 'wb'))
            break
        reward_sum = 0
        observation = env.reset()  # reset env
        prev_x = None

    if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
        print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')

# env.monitor.close() # record results for uploading.