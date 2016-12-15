
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import gym
import numpy as np
import os
import time
current_time = time.strftime("%Y%m%d_%H-%M")
flags = tf.app.flags
tf.app.flags.DEFINE_string('current_time', current_time, '')
flags.DEFINE_float('gamma', 0.99, 'discount factor')
flags.DEFINE_integer('anneal_epsilon_timesteps', 1000000, 'Number of timesteps to anneal epsilon.')
flags.DEFINE_integer('T_max', 1e+8, 'Total number of updates')
flags.DEFINE_integer('max_episode', 50000, 'Total number of updates')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags.DEFINE_string('train_dir', './tmp/' + current_time + "/", 'to store model and results.')
flags.DEFINE_string('results_dir', './tmp/', 'to store model and results.')
flags.DEFINE_float('beta_entropy', 0.01, '')  # section 8 of http://arxiv.org/pdf/1602.01783v1.pdf
tf.app.flags.DEFINE_float("eps", 1e-8, "param of avoiding probability = 0 ")
# tf.app.flags.DEFINE_string("device", "/gpu:%d" % 0, "with gpu")
tf.app.flags.DEFINE_string("device", "/cpu:%d" % 0, "with cpu")
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

def agent_model(shapes, action_types):

    with tf.device(FLAGS.device):

        state = tf.placeholder(tf.float32, shape=shapes)

        # conv1
        # hconv1 = conv2d(state, weight_variable([3, 3, shapes[-1], 16]), tf.Variable(tf.zeros(16)))
        # hpool1 = tf.nn.max_pool(hconv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')
        # hconv1 = conv2d(state, weight_variable([2, 1, shapes[-1], 16]), tf.Variable(tf.zeros(16)), strides=4)

        # hconv2 = conv2d(hpool1, weight_variable([3, 3, 16, 32]), tf.Variable(tf.zeros(32)))
        # hpool2 = tf.nn.max_pool(hconv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        h_poolf = state
        pool_shape = h_poolf.get_shape().as_list()
        h_pool2_flat = tf.reshape(h_poolf,
                                  [-1, pool_shape[1] * pool_shape[2] * pool_shape[3]])

        # fully-connect 1.
        out_channel_f = 64
        fc1 = tf.nn.bias_add(tf.matmul(h_pool2_flat,
                             weight_variable([pool_shape[1] * pool_shape[2] * pool_shape[3], out_channel_f], 1.0/np.sqrt(float(out_channel_f)))),
                             tf.Variable(tf.zeros(out_channel_f)))

        logits = tf.nn.bias_add(tf.matmul(fc1, weight_variable([out_channel_f, action_types], 1.0/np.sqrt(float(action_types)))),
                                tf.Variable(tf.zeros(action_types)))

        Q_network = tf.nn.softmax(logits)
        Value_net = tf.nn.bias_add(tf.matmul(fc1, weight_variable([out_channel_f, 1], 1.0)),
                                   tf.Variable(tf.zeros(1)))

    return Q_network, Value_net, state

def get_loss(Q_network, Value_net, num_actions):

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    R_t_placeholder = tf.placeholder("float", [None, 1])
    a_t_placeholder = tf.placeholder(tf.int32, [None])
    log_prob = - tf.reshape(tf.reduce_sum(tf.log(Q_network + FLAGS.eps) *
                                        tf.reshape(tf.one_hot(a_t_placeholder,  depth=num_actions), [-1, num_actions])
                                        , reduction_indices=1), [-1, 1]) ## -1 can also be used to infer the shape

    entropy = tf.reshape(-tf.reduce_sum(Q_network * tf.log(Q_network + FLAGS.eps), reduction_indices=1),
                         [-1, 1])  # entropy regularization

    policy_loss = log_prob * (R_t_placeholder - Value_net) + FLAGS.beta_entropy * entropy #
    value_loss = tf.nn.l2_loss(R_t_placeholder - Value_net)



    total_loss = tf.reduce_sum(policy_loss + 0.5 * value_loss, reduction_indices=0)

    train_op = optimizer.minimize(total_loss)

    return train_op, R_t_placeholder, a_t_placeholder, total_loss #, value_loss, policy_loss, entropy

def obervation2states(states, batch_size):
    # state = prepro(observation) #pong: overvation: 210x160x3 -> state: 80x80
    inchannel = 1
    shapes = (batch_size, states[0].shape[0], 1, inchannel)
    state_prep = np.reshape(states, shapes)

    return state_prep

def main(argv):
    print("\n" + FLAGS.current_time + "\n")

    episode_number = 0
    env = gym.make("CartPole-v0")
    # env.monitor.start('./results/Pong-experiment-2')  # record results for uploading

    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)

    env.reset()
    init_action = env.action_space.sample() # random walk at beginning
    num_actions = env.action_space.n
    action_space = list(range(num_actions))  # action space = Discrete(n) = [0,1,2,...n-1]

    observation, reward, done, info = env.step(init_action)

    t_max = 50


    inchannel = 1
    running_reward = None
    state = np.array(observation)  # pong: overvation: 210x160x3 -> state: 80x80
    shapes = (None, state.shape[0], 1, inchannel)
    # state_prep = np.reshape(state, shapes)


    # a = state - state_prep[0, :, :, 0] # == 0
    with tf.Graph().as_default():
        start_time = time.time()
        # define graph
        Q_network, Value_net, state_placeholder = agent_model(shapes, len(action_space))
        train_op, R_t_placeholder, a_t_placeholder, total_loss = get_loss(Q_network, Value_net, num_actions)

        saver = tf.train.Saver()

        # init variable
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

        episodes_reward_all = []
        running_reward_all = []

        ckpt_file = FLAGS.results_dir + 'NA'
        if os.path.exists(ckpt_file):
            print("\n model loaded\n")
            saver.restore(sess, ckpt_file)
            episodes_reward_all = np.load(FLAGS.results_dir + "results.npy").tolist()


        T = 0
        episode_reward = 0
        while T < FLAGS.T_max:
            t = 0
            t_start = t
            R_temp = []
            actions_temp = []
            state_temp = []

            while not (done or (t - t_start == t_max)):

                Q_prop = sess.run(Q_network, feed_dict={state_placeholder: np.reshape(state, (-1, shapes[1], shapes[2], inchannel))})
                action_index = np.argmax(Q_prop[-1])

                observation_t, reward_t, done, info = env.step(action_index)

                R_temp.append(reward_t)
                actions_temp.append(action_index)
                state_temp.append(state)
                state_new = np.array(observation)

                t += 1
                T += 1
                episode_reward += reward_t
                state = state_new - state
                # print np.sum(state_new[:])

            if done:
                R_t = 0
            else:
                R_t_temp = sess.run(Value_net, feed_dict={state_placeholder: np.reshape(state, (-1, shapes[1], shapes[2], inchannel))})
                R_t = R_t_temp[-1]

            R_batch = []
            for i in np.arange(t_start, t)[::-1]:
                R_t = R_temp[i] + FLAGS.gamma * R_t
                R_batch.append(R_t)

            _, lossval = sess.run([train_op, total_loss], feed_dict={state_placeholder: obervation2states(state_temp, -1),
                                          R_t_placeholder: np.reshape(np.array(R_batch), [-1,1]),
                                          a_t_placeholder: actions_temp})



            if done:
                episode_number += 1

                running_reward = episode_reward if running_reward is None else running_reward * 0.99 + episode_reward * 0.01

                episodes_reward_all.append(episode_reward)
                running_reward_all.append(running_reward)

                env.reset()
                init_action = env.action_space.sample()
                observation, reward, done, info = env.step(init_action)
                state = np.array(observation)
                start_time = time.time()
                if episode_number % 100 == 0:
                    print "number of episode: ", episode_number, "loss", lossval[0], "average iterations", float(T)/episode_number, \
                      "time", time.time() - start_time, "epsiode reward: ", episode_reward, "running mean: ", running_reward

                episode_reward = 0

            if (episode_number+1) % 100 == 0:
                checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
                saver.save(sess, checkpoint_file, global_step=episode_number)
                np.save(FLAGS.train_dir + "results.npy", tuple([episodes_reward_all,running_reward_all]))
                fig = plt.figure(3)
                plt.plot(episodes_reward_all, 'g--', label='rewards')
                plt.plot(running_reward_all, 'o', label='rewards')
                plt.savefig(FLAGS.train_dir + FLAGS.current_time + "rewardsplot" + '.png')
                plt.close(fig)

            if episode_number == FLAGS.max_episode:
                break



    # env.monitor.close() # record results for uploading.

if __name__=="__main__":
    tf.app.run()