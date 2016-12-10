

from keras.layers import Convolution2D
from keras.models import Model
from keras.layers import Input, Dense
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import random
import tensorflow as tf
import gym
import numpy as np

flags = tf.app.flags
flags.DEFINE_float('gamma', 0.99, 'discount factor')
flags.DEFINE_integer('anneal_epsilon_timesteps', 1000000, 'Number of timesteps to anneal epsilon.')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags.DEFINE_float('beta_entropy', 0.01, '')  # section 8 of http://arxiv.org/pdf/1602.01783v1.pdf
tf.app.flags.DEFINE_float("eps", 1e-8, "param of avoiding probability = 0 ")
FLAGS = flags.FLAGS
T = 0


def sample_final_epsilon():
    """ copy from https://github.com/coreylynch/async-rl
    Sample a final epsilon value to anneal towards from a distribution.
    These values are specified in section 5.1 of http://arxiv.org/pdf/1602.01783v1.pdf
    """
    final_epsilons = np.array([.1,.01,.5])
    probabilities = np.array([0.4,0.3,0.3])
    return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop the original image to 160x160x3
    I = I[::2, ::2, 0]  # downsample by factor of 2    from 160x160x3 to 80x80, get the first slice of tensor
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float)

def conv2d(x,W,b,strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def weight_variable(shape, std=0.1):
    initial = tf.truncated_normal(shape, stddev=std)
    return tf.Variable(initial)

def agent_model(shapes, action_types, batch_size):


    state = tf.placeholder(tf.float32, shape=shapes)

    # conv1
    hconv1 = conv2d(state, weight_variable([3,3, shapes[-1], 32]),
                    tf.Variable(tf.zeros(32)))
    hpool1 = tf.nn.max_pool(hconv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')

    # conv2
    hconv2 = conv2d(hpool1, weight_variable([3,3, 32, 64]), tf.Variable(tf.zeros(64)))
    hpool2 = tf.nn.max_pool(hconv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    h_poolf = hpool2
    pool_shape = h_poolf.get_shape().as_list()
    h_pool2_flat = tf.reshape(h_poolf,
                              [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

    # fully-connect 1.
    out_channel_f = 128
    fc1 = tf.nn.bias_add(tf.matmul(h_pool2_flat,
                         weight_variable([pool_shape[1] * pool_shape[2] * pool_shape[3], out_channel_f], 1.0/np.sqrt(float(out_channel_f)))),
                         tf.Variable(tf.zeros(out_channel_f)))

    logits = tf.nn.bias_add(tf.matmul(fc1, weight_variable([out_channel_f, action_types], 1.0/np.sqrt(float(action_types)))),
                            tf.Variable(tf.zeros(action_types)))

    Q_network = tf.nn.softmax(logits)
    Value_net = tf.nn.bias_add(tf.matmul(fc1, weight_variable([out_channel_f, batch_size], 1.0/np.sqrt(float(batch_size)))),
                               tf.Variable(tf.zeros(batch_size)))

    return Q_network, Value_net, state

def get_loss(Q_network, Value_net, batch_size, num_actions):

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    R_t_placeholder = tf.placeholder("float", [batch_size])
    a_t_placeholder = tf.placeholder(tf.int32, [batch_size])
    log_prob = tf.log(tf.reduce_sum(tf.mul(Q_network, tf.one_hot(a_t_placeholder,  depth=num_actions)), reduction_indices=1))

    policy_loss = - log_prob * (R_t_placeholder - Value_net)
    value_loss = tf.reduce_mean(tf.square(R_t_placeholder - Value_net))

    entropy = - tf.reduce_sum(Q_network * tf.log(Q_network + FLAGS.eps))  # entropy regularization

    total_loss = policy_loss + (0.5 * value_loss) + FLAGS.beta_entropy * entropy

    train_op = optimizer.minimize(total_loss)

    return train_op, R_t_placeholder, a_t_placeholder

def obervation2states(states, batch_size):
    # state = prepro(observation) #pong: overvation: 210x160x3 -> state: 80x80
    inchannel = 1
    shapes = (batch_size, states[0].shape[0], states[0].shape[1], inchannel)
    state_prep = np.reshape(states, shapes)

    return state_prep

def main(argv):

    episode_number = 0
    epsilon = 1
    T_max = 1000
    env = gym.make("Pong-v0")


    init_action = env.action_space.sample() # random walk at beginning
    num_actions = env.action_space.n
    action_space = list(range(num_actions))  # action space = Discrete(n) = [0,1,2,...n-1]

    observation, reward, done, info = env.step(init_action)


    t_max = 2
    batch_size = t_max

    inchannel = 1
    state = prepro(observation)  # pong: overvation: 210x160x3 -> state: 80x80
    shapes = (batch_size, state.shape[0], state.shape[1], inchannel)
    # state_prep = np.reshape(state, shapes)


    # a = state - state_prep[0, :, :, 0] # == 0
    with tf.Graph().as_default():

        # define graph
        Q_network, Value_net, state_placeholder = agent_model(shapes, len(action_space), batch_size)
        train_op,R_t_placeholder, a_t_placeholder = get_loss(Q_network, Value_net, batch_size, num_actions)

        # init variable
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)



        T = 0

        while T < T_max:
            t = 0
            t_start = t
            R_temp = []
            actions_temp = []
            state_temp = []

            while not (done or (t - t_start == t_max)):
                Q_prop = sess.run(Q_network, feed_dict={state_placeholder: np.reshape([state] * batch_size, shapes)})[0]
                action_index = np.argmax(Q_prop[0])  # first batch

                observation_t, reward_t, done, info = env.step(action_index)

                R_temp.append(reward_t)
                actions_temp.append(action_index)
                state_temp.append(prepro(observation))
                t += 1
                T += 1

                state = prepro(observation)

            if done:
                R_t = 0
            else:
                R_t = sess.run(Value_net, feed_dict={state_placeholder: np.reshape([state] * batch_size, shapes)})[0][0]

            R_batch = np.zeros(t)
            for i in np.arange(t_start, t)[::-1]:
                R_t = R_temp[i] + FLAGS.gamma * R_t
                R_batch[i] = R_t

            sess.run(train_op, feed_dict={state_placeholder: obervation2states(state_temp, batch_size),
                                          R_t_placeholder: R_batch,
                                          a_t_placeholder: actions_temp})




if __name__=="__main__":
    tf.app.run()