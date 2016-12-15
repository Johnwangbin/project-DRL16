'''
Baseline of A3C model.
Combine the idea from (environment): https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-2-ded33892c724#.mtwpvfi8b
and (A3C network) https://github.com/coreylynch/async-rl/blob/master/a3c_model.py
'''

import numpy as np
import tensorflow as tf
import gym
import time

env = gym.make("Pong-v0")

# hyperparameters
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4 # feel free to play with this to train faster or more stably.
gamma = 0.99 # discount factor for reward

tf.reset_default_graph()

with tf.device("/gpu:0"): #this is a must, when run on GPU: "/gpu:0", on CPU: "/cpu:0"
    observations = tf.placeholder(tf.float32, [None, 80, 80, 1] , name="input_x")
    W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 1, 16], stddev=0.01))
    conv1 = tf.nn.conv2d(observations, W_conv1, strides=[1, 1, 1, 1], padding='SAME', name="conv1") # Conv layer
    h_conv1 = tf.nn.relu(conv1) # Relu layer
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME') # h_pool1 is (?, 20, 20, 16)

    W_conv2 = tf.Variable(tf.truncated_normal([4, 4, 16, 32], stddev=0.01))
    conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME', name="conv1")
    h_conv2 = tf.nn.relu(conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # (?, 10, 10, 32)

    h_f = tf.contrib.layers.flatten(h_pool2) # flatten for dense layer, 3200

    W_d = tf.Variable(tf.truncated_normal([3200, 256], stddev=0.01))
    h_d = tf.nn.relu(tf.matmul(h_f, W_d)) # dense layer

    W_o = tf.Variable(tf.truncated_normal([256, 2], stddev=0.01))
    probability = tf.nn.softmax(tf.matmul(h_d, W_o)) # output layer, Tensor("Softmax:0", shape=(?, 2), dtype=float32)


#From here we define the parts of the network needed for learning a good policy.
tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32,[None,1], name="input_y")
advantages = tf.placeholder(tf.float32,name="reward_signal")

# The loss function. This sends the weights in the direction of making actions
# that gave good advantage (reward over time) more likely, and actions that didn't less likely.
loglik = tf.log(input_y*(input_y - probability) + (1 - input_y)*(input_y + probability))
loss = -tf.reduce_mean(loglik * advantages)
newGrads = tf.gradients(loss,tvars)

# Once we have collected a series of gradients from multiple episodes, we apply them.
# We don't just apply gradeients after every episode in order to account for noise in the reward signal.
adam = tf.train.AdamOptimizer(learning_rate=learning_rate) # Our optimizer
W1Grad = tf.placeholder(tf.float32,name="batch_grad1") # Placeholders to send the final gradients through when we update.
W2Grad = tf.placeholder(tf.float32,name="batch_grad2")
batchGrad = [W1Grad,W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad,tvars))

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
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float) # ravel is to flatten, so no ravel here

xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 1
total_episodes = 8000
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    rendering = False
    sess.run(init)
    observation = env.reset()  # Obtain an initial observation of the environment

    # Reset the gradient placeholder. We will collect gradients in
    # gradBuffer until we are ready to update our policy network.
    gradBuffer = sess.run(tvars)
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while episode_number <= total_episodes:
        start_time = time.time()
        # Rendering the environment slows things down,
        # so let's only look at it once our agent is doing a good job.
        if reward_sum / batch_size > 100 or rendering == True:
            env.render()
            rendering = True

        x = prepro(observation) # preprocessing to appropriate shape
        x = tf.reshape(x, [1, 80, 80, 1]) # batch and channel axisension, (1, 80, 80, 1)
        x = x.eval() # <type 'numpy.ndarray'>
        #x = tf.expand_dims(tf.expand_dims(tf.Variable(x), 0), 3) # batch and channel axisension, (1, 80, 80, 1)

        # Run the policy network and get an action to take.
        tfprob = sess.run(probability, feed_dict={observations: x}) # type of tfprob is numpy array
        action = 1 if np.random.uniform() < tfprob[0][1] else 0

        xs.append(x)  # observation
        y = 1 if action == 0 else 0  # a "fake label"
        ys.append(y)

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward

        drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

        if done:

            episode_number += 1
            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            tfp = tfps
            xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []  # reset array memory

            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(epr)
            # size the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            # Get the gradient for this episode, and save it in the gradBuffer
            tGrad = sess.run(newGrads, feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
            for ix, grad in enumerate(tGrad):
                gradBuffer[ix] += grad

            # If we have completed enough episodes, then update the policy network with our gradients.
            if episode_number % batch_size == 0:
                sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

                # Give a summary of how well our network is doing for each batch of episodes.
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print 'Reward for episode %f.  Total reward %f. running time: %f secs' % (
                reward_sum, running_reward, time.time() - start_time)
            start_time = 0


            if reward_sum / batch_size > 200:
                print "Task solved in", episode_number, 'episodes!'
                break

            reward_sum = 0

            observation = env.reset()

print episode_number, 'Episodes completed.'