#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 16:28:39 2018

@author: minhan
"""

import tensorflow as tf
import numpy as np
import gym
import tensorflow.contrib.slim as slim

import sys
if "../" not in sys.path:
    sys.path.append("../") 
    
from lib.running_variance import RunningVariance
from lib import plotting

np.random.seed(123)

env = gym.make('CartPole-v0')

state_dim = env.observation_space.shape[0] 
action_count = env.action_space.n 
hidden_size = 128 
update_frequency = 20

class agent():
    def __init__(self, lr):
        self.observations= tf.placeholder(shape=[None,state_dim],dtype=tf.float32)
        hidden = slim.fully_connected(self.observations,hidden_size,biases_initializer=None,activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden,action_count,activation_fn=tf.nn.softmax,biases_initializer=None)
        self.chosen_action = tf.argmax(self.output,1)

        self.return_weight = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
        
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.return_weight)
        
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        
        self.gradients = tf.gradients(self.loss,tvars)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))


def discount_rewards(r, gamma = 0.999):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

tf.reset_default_graph()

myAgent = agent(lr=1e-2,)
max_number_of_episodes = 1000
running_variance = RunningVariance()
reward_sum = 0

stats = plotting.EpisodeStats(
    episode_lengths=np.zeros(max_number_of_episodes),
    episode_rewards=np.zeros(max_number_of_episodes),
    episode_running_variance=np.zeros(max_number_of_episodes))   

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    total_reward = []
    total_length = []
        
    gradBuffer = sess.run(tf.trainable_variables())
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
        
    for episode_number in range(max_number_of_episodes):
        s = env.reset()
        running_reward = 0
        ep_history = []
        done = False
        t = 1
        while not done:
            
            a_dist = sess.run(myAgent.output,feed_dict={myAgent.observations:[s]})
            a = np.random.choice(a_dist[0],p=a_dist[0])
            a = np.argmax(a_dist == a)
            
            env.render()
            
            ns,r,done,_ = env.step(a) 
            ep_history.append([s,a,r,ns])
            s = ns
            
            running_reward += float(r)
                    
            stats.episode_rewards[episode_number] += r
            stats.episode_lengths[episode_number] = t
        
            t += 1


        ep_history = np.array(ep_history)
        ep_history[:,2] = discount_rewards(ep_history[:,2])
        
        for r in ep_history[:,2]:    
            running_variance.add(r)
       
        feed_dict={myAgent.return_weight:ep_history[:,2],
                   myAgent.action_holder:ep_history[:,1],
                   myAgent.observations:np.vstack(ep_history[:,0])}
        
        grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
        
        for idx,grad in enumerate(grads):
            gradBuffer[idx] += grad

        if episode_number % update_frequency == 0 and episode_number != 0:
            feed_dict = dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
            _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
            for ix,grad in enumerate(gradBuffer):
                gradBuffer[ix] = grad * 0
            print('Episode: %d. Average reward for episode %f. Variance %f' % (episode_number, np.mean(total_reward[-update_frequency:]), running_variance.get_variance()))

        total_reward.append(running_reward)
        total_length.append(t)
        
plotting.plot_pgresults(stats)
