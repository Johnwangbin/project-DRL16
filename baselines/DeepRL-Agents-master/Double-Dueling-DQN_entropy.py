
# coding: utf-8

# # Simple Reinforcement Learning with Tensorflow Part 4: Deep Q-Networks and Beyond
# 
# In this iPython notebook I implement a Deep Q-Network using both Double DQN and Dueling DQN. The agent learn to solve a navigation task in a basic grid world. To learn more, read here: https://medium.com/p/8438a3e2b8df
# 
# For more reinforcment learning tutorials, see:
# https://github.com/awjuliani/DeepRL-Agents

# In[ ]:

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import gym
import numpy as np
import random
import tensorflow as tf

import scipy.misc
import os

current_time = time.strftime("%Y%m%d_%H-%M")
flags = tf.app.flags
tf.app.flags.DEFINE_string('current_time', current_time, '')
flags.DEFINE_string('train_dir', './tmp/' + current_time + "/", 'to store model and results.')
flags.DEFINE_string('results_dir', './tmp/', 'to store model and results.')
flags.DEFINE_float('beta_entropy', 0.01, '')  # section 8 of http://arxiv.org/pdf/1602.01783v1.pdf
tf.app.flags.DEFINE_float("eps", 1e-8, "param of avoiding probability = 0 ")
tf.app.flags.DEFINE_string("device", "/gpu:%d" % 0, "with gpu")
# tf.app.flags.DEFINE_string("device", "/cpu:%d" % 0, "with cpu")
FLAGS = flags.FLAGS


env = gym.make("Pong-v0")


# Above is an example of a starting environment in our simple game. The agent controls the blue square, and can move up, down, left, or right. The goal is to move to the green square (for +1 reward) and avoid the red square (for -1 reward). The position of the three blocks is randomized every episode.

# ### Implementing the network itself


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop the original image to 160x160x3
    I = I[::2, ::2, 0]  # downsample by factor of 2    from 160x160x3 to 80x80, get the first slice of tensor
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1

    return I.astype(np.float).ravel()

class Qnetwork():
    def __init__(self,h_size):
        #The network recieves a frame from the game, flattened into an array.
        #It then resizes it and processes it through four convolutional layers.

        self.scalarInput = tf.placeholder(shape=[None,80*80],dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput,shape=[-1,80,80,1])
        self.conv1 = tf.contrib.layers.convolution2d(inputs=self.imageIn,num_outputs=32,kernel_size=[8,8],stride=[4,4],padding='VALID', biases_initializer=None)
        self.conv2 = tf.contrib.layers.convolution2d(inputs=self.conv1,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID', biases_initializer=None)
        self.conv3 = tf.contrib.layers.convolution2d(inputs=self.conv2,num_outputs=64,kernel_size=[3,3],stride=[1,1],padding='VALID', biases_initializer=None)
        self.conv4 = tf.contrib.layers.convolution2d(inputs=self.conv3,num_outputs=512,kernel_size=[3,3],stride=[4,4],padding='VALID', biases_initializer=None)
        
        #We take the output from the final convolutional layer and split it into separate advantage and value streams.
        self.streamAC, self.streamVC = tf.split(3,2,self.conv4)
        self.streamA = tf.contrib.layers.flatten(self.streamAC)
        self.streamV = tf.contrib.layers.flatten(self.streamVC)
        # self.AW = tf.Variable(tf.random_normal([h_size/2,env.actions]))
        self.AW = tf.Variable(tf.random_normal([h_size/2,env.action_space.n]))
        self.VW = tf.Variable(tf.random_normal([h_size/2,1]))
        self.Advantage = tf.matmul(self.streamA,self.AW)
        self.Value = tf.matmul(self.streamV,self.VW)
        
        #Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.sub(self.Advantage,tf.reduce_mean(self.Advantage,reduction_indices=1,keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)
        
        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        # self.actions_onehot = tf.one_hot(self.actions,env.actions,dtype=tf.float32)
        self.actions_onehot = tf.one_hot(self.actions,env.action_space.n,dtype=tf.float32)
        
        self.Q = tf.reduce_sum(tf.mul(self.Qout, self.actions_onehot), reduction_indices=1)

        self.Qprob = tf.nn.softmax(self.Qout)
        self.entropy = tf.reduce_sum(tf.reshape(
            -tf.reduce_sum(self.Qprob * tf.log(self.Qprob + FLAGS.eps), reduction_indices=1), [-1, 1]))

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error) + self.entropy * FLAGS.beta_entropy
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)


# ### Experience Replay

# This class allows us to store experies and sample then randomly to train the network.

# In[ ]:

class experience_buffer():
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)
            
    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])


# This is a simple function to resize our game frames.

# In[ ]:

def processState(states):
    return np.reshape(states,[80*80])


# These functions allow us to update the parameters of our target network with those of the primary network.

# In[ ]:

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars/2]):
        op_holder.append(tfVars[idx+total_vars/2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars/2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)


# ### Training the network

# Setting all the training parameters
batch_size = 32  #How many experiences to use for each training step.
update_freq = 4 #How often to perform a training step.
y = .99 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action
endE = 0.1 #Final chance of random action
anneling_steps = 10000. #How many steps of training to reduce startE to endE.
num_episodes = 10000 #How many episodes of game environment to train network with.
pre_train_steps = 10000 #How many steps of random actions before training begins. # 77 episodes
max_epLength = 2000 #The max allowed length of our episode.
load_model = False #Whether to load a saved model.
path = "./dqn" #The path to save our model to.
h_size = 512     #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network


# In[ ]:

tf.reset_default_graph()
mainQN = Qnetwork(h_size)
targetQN = Qnetwork(h_size)

init = tf.initialize_all_variables()

saver = tf.train.Saver()

trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables,tau)

myBuffer = experience_buffer()

#Set the rate of random action decrease. 
e = startE
stepDrop = (startE - endE)/anneling_steps

#create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0
running_reward = None

rewards_record = []
running_reward_records = []
#Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:
    if load_model == True:
        print 'Loading Model...'
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    sess.run(init)
    updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.
    for i in range(num_episodes):

        start_time = time.time()
        episodeBuffer = experience_buffer()
        #Reset environment and get first new observation
        s = env.reset()

        s = prepro(s)

        d = False
        episode_reward = 0
        j = 0
        #The Q-Network
        while j < max_epLength: #If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0,4)
            else:
                a = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:[s]})[0]
            s1, rewards_curr, d, info = env.step(a)
            s1 = prepro(s1)
            # s1 = processState(s1)


            total_steps += 1
            episodeBuffer.add(np.reshape(np.array([s,a,rewards_curr,s1,d]),[1,5])) #Save the experience to our episode buffer.
            entropy = 0
            loss = 0
            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop
                
                if total_steps % (update_freq) == 0:
                    trainBatch = myBuffer.sample(batch_size) #Get a random batch of experiences.
                    #Below we perform the Double-DQN update to the target Q-values
                    Q1 = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3])})
                    Q2 = sess.run(targetQN.Qout,feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3])})
                    # print Q2
                    end_multiplier = -(trainBatch[:,4] - 1)
                    doubleQ = Q2[range(batch_size),Q1]
                    targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
                    #Update the network with our target values.
                    _, entropy, loss = sess.run([mainQN.updateModel, mainQN.entropy,mainQN.loss],
                                                 feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]),
                                                            mainQN.targetQ:targetQ,
                                                            mainQN.actions:trainBatch[:,1]})
                    updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.
            episode_reward += rewards_curr
            s = s1
            
            if d == True:
                running_reward = episode_reward if running_reward is None else running_reward * 0.99 + episode_reward * 0.01
                print 'resetting env. %dth episode, entropy %f, loss %f, reward total was %f. running mean: %f, %f secs' \
                      % (i, entropy, loss, episode_reward, running_reward, time.time() - start_time)
                start_time = time.time()
                rewards_record.append(episode_reward)
                running_reward_records.append(running_reward)

                break
        
        #Get all experiences from this episode and discount their rewards.
        myBuffer.add(episodeBuffer.buffer)
        jList.append(j)

        #Periodically save the model. 
        if i % 1000 == 0:
            saver.save(sess,path+'/model-'+str(i)+'.cptk')
            np.save(path, tuple([rewards_record, running_reward_records]))
            print "Saved Model"
        # if len(rList) % 10 == 0:




    # saver.save(sess,path+'/model-'+str(i)+'.cptk')
# print "Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%"


# ### Checking network learning

# Mean reward over time

# In[ ]:

rMat = np.resize(np.array(rList),[len(rList)/100,100])
rMean = np.average(rMat,1)
plt.plot(rMean)

