import tensorflow as tf
import numpy as np
from mlp.data_providers import MSD10GenreDataProvider,MSD25GenreDataProvider
import matplotlib.pyplot as plt
import time
import seaborn as sns
import pandas as pd
from providers import NoisyMSD10DataProvider,NoisyMSD25DataProvider,DropOutMSD10DataProvider,DropOutMSD25DataProvider
from tensorflow.contrib import rnn
time_steps = 120
step_dim = 25
seed = 123
batch_size = 50
rng = np.random.RandomState(seed)

class RNN_Model(object):
    
    def __init__(self,layers=2,num_hidden=200,lr=1e-3,num_epochs=10,provider=0,out=1):
        if provider == 0:
            self.train_data = MSD10GenreDataProvider('train', batch_size=batch_size, rng=rng)
            self.valid_data = MSD10GenreDataProvider('valid', batch_size=batch_size, rng=rng)
        else:
            self.train_data = MSD25GenreDataProvider('train', batch_size=batch_size, rng=rng)
            self.valid_data = MSD25GenreDataProvider('valid', batch_size=batch_size, rng=rng)
        self.inputs = tf.placeholder(tf.float32, [None, self.train_data.inputs.shape[1]], 'inputs')
        self.targets = tf.placeholder(tf.float32, [None, self.train_data.num_classes], 'targets')
        self.layers=layers
        self.num_hidden = num_hidden
        self.num_epochs = num_epochs
        self.lr = lr
        self.out = out
        self.out = out
        if provider == 0:
            self.MSD = 'RNN_MSD10 '
        else:
            self.MSD = 'RNN_MSD25 '
        self.title = self.MSD + ' LR = ' + str(self.lr)
        
       
    def RNN_layers(self,inputs,nonlinearity=tf.nn.relu):
        inpputs = inputs.reshape([50,120,25])
        
        lstm_cell = rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0)
        
    def run_session(self):
        sess = tf.Session()
        sess.run(self.init)
        self.acct = np.zeros(self.num_epochs)
        self.errt = np.zeros(self.num_epochs)
        self.accv = np.zeros(self.num_epochs)
        self.errv = np.zeros(self.num_epochs)
        self.times = np.zeros(self.num_epochs)
        for e in range(self.num_epochs):
            start_time = time.time()
            running_error = 0.
            running_accuracy = 0.
            valid_error = 0.
            valid_accuracy = 0.
            for input_batch, target_batch in self.train_data:
                _, batch_error, batch_acc = sess.run(
                    [self.train_step, self.error, self.accuracy],
                    feed_dict={self.inputs: input_batch, self.targets: target_batch})
                running_error += batch_error
                running_accuracy += batch_acc
            end_time=time.time()
            run_time=end_time-start_time
            self.times[0,e]=run_time
            running_error /= self.train_data.num_batches
            running_accuracy /= self.train_data.num_batches

            for input_batch, target_batch in self.valid_data:
                batch_error, batch_acc = sess.run(
                    [self.error, self.accuracy],
                    feed_dict={self.inputs: input_batch, self.targets: target_batch})
                valid_error += batch_error
                valid_accuracy += batch_acc
            valid_error /= self.valid_data.num_batches
            valid_accuracy /= self.valid_data.num_batches
            if self.out==1:
                print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f} run_time={3:.2f}s | err(valid)={4:.2f} acc(valid)={5:.2f}'
                      .format(e + 1, running_error, running_accuracy, run_time,valid_error, valid_accuracy))
            self.errt[e]=(running_error)
            self.errv[e]=(valid_error)
            self.acct[e]=(running_accuracy)
            self.accv[e]=(valid_accuracy)
        self.avg_time,self.min_err,self.max_acc=np.mean(self.times),np.min(self.errv),np.max(self.accv)
        if self.out==1:
            self.basic_plot()
