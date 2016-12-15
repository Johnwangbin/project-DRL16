
# coding: utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# # Simple Reinforcement Learning in Tensorflow Part 2-b: 
# ## Vanilla Policy Gradient Agent
# This tutorial contains a simple example of how to build a policy-gradient based agent that can solve the CartPole problem. For more information, see this [Medium post](https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-2-ded33892c724#.mtwpvfi8b). This implementation is generalizable to more than two actions.
# 
# For more Reinforcement Learning algorithms, including DQN and Model-based learning in Tensorflow, see my Github repo, [DeepRL-Agents](https://github.com/awjuliani/DeepRL-Agents). 

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import os
import time

current_time = time.strftime("%Y%m%d_%H-%M")
flags = tf.app.flags
tf.app.flags.DEFINE_string('current_time', current_time, '')
flags.DEFINE_string('train_dir', './tmp/' + current_time + "/", 'to store model and results.')
flags.DEFINE_string('results_dir', './tmp/' + "NA" + "/", 'to store model and results.')

# tf.app.flags.DEFINE_string("device", "/gpu:%d" % 0, "with gpu")
# tf.app.flags.DEFINE_string("device", "/cpu:%d" % 0, "with cpu")
FLAGS = flags.FLAGS




env = gym.make('Pong-v0')
# env.monitor.start('./results/Pong-experiment-2')  # record results for uploading

# ### The Policy-Based Agent
gamma = 0.99
def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop the original image to 160x160x3
    I = I[::2, ::2, 0]  # downsample by factor of 2    from 160x160x3 to 80x80, get the first slice of tensor
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

class agent():
    def __init__(self, lr, s_size,a_size,h_size):
        # with tf.device(FLAGS.device):
        #the feed-forward part of the network.
        self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in, h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=None) # 6
        self.chosen_action = tf.argmax(self.output, 1)

        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)

        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)

        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))


# ### Training the Agent
tf.reset_default_graph() #Clear the Tensorflow graph.

myAgent = agent(lr=1e-2, s_size=80*80, a_size=6, h_size=8)

total_episodes = 10000  # Set total number of episodes to train agent on.
max_ep = 10000
update_frequency = 5

saver = tf.train.Saver()

init = tf.initialize_all_variables()

if not os.path.exists(FLAGS.train_dir):
    os.makedirs(FLAGS.train_dir)


# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    i = 1
    T = 0
    total_reward = []
    total_running_reward = []
    total_lenght = []

    ckpt_file = FLAGS.results_dir + 'checkpoint-9000'
    if os.path.exists(ckpt_file):
        print("\n model loaded\n")
        saver.restore(sess, ckpt_file)
        total_reward, total_running_reward= np.load(FLAGS.results_dir + "results.npy").tolist()

    gradBuffer = sess.run(tf.trainable_variables())
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    running_reward = None
    while i < total_episodes:
        start_time = time.time()
        s = prepro(env.reset())
        episode_reward = 0
        ep_history = []
        for j in range(max_ep):
            T += 1
            # random sample according to the policy net output.
            a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in:[s]})
            a = np.random.choice(np.arange(env.action_space.n), p=a_dist[0])

            s1, r, d, info = env.step(a) #Get our reward for taking an action given a bandit.
            ep_history.append([s,a,r,s1])
            s = prepro(s1)
            episode_reward += r
            if d == True:
                #Update the network.
                ep_history = np.array(ep_history)
                ep_history[:, 2] = discount_rewards(ep_history[:,2])  # replace reward with discounted one
                feed_dict={myAgent.reward_holder:ep_history[:,2],
                        myAgent.action_holder:ep_history[:,1], myAgent.state_in:np.vstack(ep_history[:,0])}
                grads, lossval = sess.run([myAgent.gradients, myAgent.loss], feed_dict=feed_dict)
                for idx,grad in enumerate(grads):
                    gradBuffer[idx] += grad

                if i % update_frequency == 0 and i != 0:
                    feed_dict= dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                    _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)

                    for ix,grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0

                # lkx
                grads2 = sess.run(myAgent.gradients, feed_dict=feed_dict)
                print np.array(grads2[0]) - np.array(grads[0])

                
                total_reward.append(episode_reward)
                total_lenght.append(j)

                episode_number = i
                running_reward = episode_reward if running_reward is None else running_reward * 0.99 + episode_reward * 0.01
                print "number of episode: ", episode_number, "loss", lossval, "average iterations", float(T)/episode_number, \
                      "time", time.time() - start_time, "epsiode reward: ", episode_reward, "running mean: ", running_reward
                total_running_reward.append(running_reward)

                break

        i += 1

        if i % 1000 == 0:
            # save model
            checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
            saver.save(sess, checkpoint_file, global_step=i)
            np.save(FLAGS.train_dir + "results.npy", tuple([total_reward, total_running_reward]))

            fig = plt.figure(3)
            plt.plot(total_reward, 'g--', label='rewards')
            plt.plot(total_running_reward, 'o', label='running_mean_reward')
            plt.savefig(FLAGS.train_dir + FLAGS.current_time + "rewardsplot" + '.png')
            plt.close(fig)

env.monitor.close() # record results for uploading.
