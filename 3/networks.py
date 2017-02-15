import os
import tensorflow as tf
import numpy as np
from mlp.data_providers import MSD10GenreDataProvider,MSD25GenreDataProvider
import matplotlib.pyplot as plt
import time
import seaborn as sns
import pandas as pd
from providers import AugmentedMSD10DataProvider,AugmentedMSD25DataProvider
import pickle
seed = 123
rng = np.random.RandomState(seed)

class Model(object):
    
    def __init__(self,layers=2,num_hidden=200,lr=1e-3,num_epochs=10,provider=0,out=1):
        if provider == 0:
            self.train_data = MSD10GenreDataProvider('train', batch_size=50, rng=rng)
            self.valid_data = MSD10GenreDataProvider('valid', batch_size=50, rng=rng)
        else:
            self.train_data = MSD25GenreDataProvider('train', batch_size=50, rng=rng)
            self.valid_data = MSD25GenreDataProvider('valid', batch_size=50, rng=rng)
        self.inputs = tf.placeholder(tf.float32, [None, self.train_data.inputs.shape[1]], 'inputs')
        self.targets = tf.placeholder(tf.float32, [None, self.train_data.num_classes], 'targets')
        self.layers=layers
        self.num_hidden = num_hidden
        self.num_epochs = num_epochs
        self.lr = lr
        self.out = out
        if provider == 0:
            self.MSD = 'MSD10 '
        else:
            self.MSD = 'MSD25 '
        self.title = self.MSD + 'N = ' + str(self.num_hidden) + ', L = ' + str(self.layers) +', LR = ' + str(self.lr)
        with tf.name_scope('fc-layer-1'):
            hidden_1 = self.fully_connected_layer(self.inputs,self.train_data.inputs.shape[1],self.num_hidden)
        if layers>1:
            with tf.name_scope('fc-h'):
                hiddens = self.hidden_layers(hidden_1,self.num_hidden,n=layers-1,nonlinearity=tf.nn.relu)
            with tf.name_scope('output-layer'):
                self.outputs = self.fully_connected_layer(hiddens, self.num_hidden, self.train_data.num_classes, tf.identity)
        else:
            with tf.name_scope('output-layer'):
                self.outputs = self.fully_connected_layer(hidden_1, self.num_hidden, self.train_data.num_classes, tf.identity)            
    
    def run_session(self):
        sess = tf.Session()
        sess.run(self.init)
        self.acct = np.zeros([1,self.num_epochs])
        self.errt = np.zeros([1,self.num_epochs])
        self.accv = np.zeros([1,self.num_epochs])
        self.errv = np.zeros([1,self.num_epochs])
        times = np.zeros([1,self.num_epochs])
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
            times[0,e]=run_time
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
            self.errt[0,e]=(running_error)
            self.errv[0,e]=(valid_error)
            self.acct[0,e]=(running_accuracy)
            self.accv[0,e]=(valid_accuracy)
        self.avg_time,self.min_err,self.max_acc=np.mean(times),np.min(self.errv),np.max(self.accv)
        if self.out==1:
            self.basic_plot()
            
    def basic_plot(self):
            fig, (ax_1, ax_2) = plt.subplots(1, 2,figsize=(10,4))
            print('{0:s} Done! Avg. Epoch Time: {1:.2f}s, Best Val. Error: {2:.2f}, Best Val. Accuracy: {3:.2f}'.format(
                    self.title,self.avg_time,self.min_err,self.max_acc))
            for d,k in zip([self.errt[0,:],self.errv[0,:]],['err(train)', 'err(valid)']):
                ax_1.plot(np.arange(1, self.num_epochs+1), 
                          d, label=k)
            ax_1.set_ylabel('error',visible=True)
            ax_1.set_xlabel('epoch')
            ax_1.legend(loc=0)
            
            for d,k in zip([self.acct[0,:],self.accv[0,:]],['acc(train)', 'acc(valid)']):
                ax_2.plot(np.arange(1, self.num_epochs+1), 
                          d, label=k)
            ax_2.set_xlabel('epoch')
            ax_2.set_ylabel('accuracy')
            ax_2.legend(loc=0)
            fig.suptitle(self.title)
    
    def fully_connected_layer(self,inputs,input_dim,output_dim,nonlinearity=tf.nn.relu):
        weights = tf.Variable(
            tf.truncated_normal(
                [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5,seed=123), 
            'weights')
        biases = tf.Variable(tf.zeros([output_dim]), 'biases')
        outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
        return outputs

    def hidden_layers(self,inputs,hidden_dim,n=1,nonlinearity=tf.nn.relu):
        weights = tf.Variable(
            tf.truncated_normal(
                [hidden_dim, hidden_dim], stddev=2. / (hidden_dim*2)**0.5,seed=123), 
            'weights')
        
        biases = tf.Variable(tf.zeros([hidden_dim]), 'biases')
        outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
        hiddens = tf.contrib.slim.stack(outputs, tf.contrib.slim.fully_connected,[self.num_hidden]*n)
        return hiddens

class SimpleModel(Model):
    
    def __init__(self,layers=1,num_hidden=200,lr=1e-4,num_epochs=10,provider=0,out=1):
        super().__init__(layers,num_hidden,lr,num_epochs,provider,out)
        self.learning_functions()

    def learning_functions(self):
        with tf.name_scope('error'):
            self.error = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(self.outputs, self.targets))
        with tf.name_scope('accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(
                    tf.equal(tf.argmax(self.outputs, 1), tf.argmax(self.targets, 1)), 
                    tf.float32))
        with tf.name_scope('train'):
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.error)
            
        self.init = tf.global_variables_initializer()
        
class AugmentedSimpleModel(SimpleModel):
    
    def __init__(self,layers=1,num_hidden=200,lr=1e-4,num_epochs=10,provider=0,out=1,
                 frac=0.15,std=0.05):
        super().__init__(layers,num_hidden,lr,num_epochs,provider,out)
        self.frac=frac
        self.std=std
        if self.provider == 0:
            self.train_data = AugmentedMSD10DataProvider('train', batch_size=50, rng=rng,frac=self.frac,std=self.std)
        else:
            self.train_data = AugmentedMSD25DataProvider('train', batch_size=50, rng=rng,frac=self.frac,std=self.std)

class RegModel(Model):
    
    def __init__(self,layers=1,num_hidden=200,lr=1e-4,num_epochs=10,provider=0,out=1,
                 reg=1,rc=1e-3):
        super().__init__(layers,num_hidden,lr,num_epochs,provider,out)
        self.reg = reg
        self.rc = rc
        self.learning_functions()
        if reg == 1:
            self.title = self.MSD + 'L2 Coeff = ' + str(self.rc)
        else:
            self.title = self.MSD + 'L1 Coeff = ' + str(self.rc)
    
    def learning_functions(self):
        vars   = tf.trainable_variables()
        if self.reg==1:#l2 loss
            with tf.name_scope('error'):
                regloss = tf.add_n([ tf.nn.l2_loss(v) for v in vars
                                   if 'biases' not in v.name ]) * self.rc
        else:#l1 loss
            with tf.name_scope('error'):
                regloss = tf.add_n([ tf.reduce_sum(tf.abs(v)) for v in vars
                                   if 'biases' not in v.name ]) * self.rc
        self.error = tf.reduce_mean(regloss +
            tf.nn.softmax_cross_entropy_with_logits(self.outputs, self.targets))
        with tf.name_scope('accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(
                tf.equal(tf.argmax(self.outputs, 1), tf.argmax(self.targets, 1)), 
                tf.float32))
        with tf.name_scope('train'):
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.error)
            
        self.init = tf.global_variables_initializer()
        
class AugmentedRegModel(RegModel):
    
    def __init__(self,layers=1,num_hidden=200,lr=1e-3,num_epochs=10,provider=0,out=1,
                 reg=1,rc=1e-3,fraction=0.15,std=0.01):
        super().__init__(layers,num_hidden,lr,num_epochs,provider,out,reg,rc)
        self.fraction=fraction
        self.std=std
        self.title = self.title + ', NL = ' + str(self.std) + ', AL = ' + str(self.fraction*100) + '%'
        if provider == 0:
            self.train_data = AugmentedMSD10DataProvider('train', batch_size=50, rng=rng,fraction=self.fraction,std=self.std)
        else:
            self.train_data = AugmentedMSD25DataProvider('train', batch_size=50, rng=rng,fraction=self.fraction,std=self.std)
        
class MultiPlot(object):
    def __init__(self,sims,labels):
        self.sims = sims
        self.d1 = len(sims[0][:])
        self.d2 = len(sims[:][0])
        self.labels = labels
        self.times = np.zeros([self.d1,self.d2])
        self.errs = np.zeros([self.d1,self.d2])
        self.accs = np.zeros([self.d1,self.d2])
        for k in range(self.d1):
            for j in range(self.d2):
                self.times[k][j],self.errs[k][j],self.accs[k][j]=self.sims[k][j].avg_time,np.min(self.sims[k][j].errv),np.max(self.sims[k][j].accv)
    
    def err_grid(self):
        fig, axarr = plt.subplots(self.d1, self.d2,figsize=(16,8))
        for k in range(self.d1):
            axarr[k,0].set_ylabel('error')
            for j in range(self.d2):
                for d,w in zip([self.sims[k][j].errt[0,:],self.sims[k][j].errv[0,:]],['err(train)', 'err(valid)']):
                    axarr[k,j].plot(np.arange(1, self.sims[k][j].num_epochs+1), 
                              d, label=w)
                if k!=self.d1-1:
                    plt.setp(axarr[k,j].get_xticklabels(), visible=False)
                else:
                    axarr[k][j].set_xlabel('epoch')
        axarr[0,0].legend(loc=0)
    
    def acc_grid(self):
        fig, axarr = plt.subplots(self.d1, self.d2,figsize=(16,8))
        for k in range(self.d1):
            axarr[k,0].set_ylabel('accuracy')
            for j in range(self.d2):
                for d,w in zip([self.sims[k][j].acct[0,:],self.sims[k][j].accv[0,:]],['acc(train)', 'acc(valid)']):
                    axarr[k,j].plot(np.arange(1, self.sims[k][j].num_epochs+1), 
                              d, label=w)
                if k!=self.d1-1:
                    plt.setp(axarr[k,j].get_xticklabels(), visible=False)
                else:
                    axarr[k][j].set_xlabel('epoch')
            
        axarr[0,0].legend(loc=0)
        
    def time_heat(self,rs,cs):
        rs = [str(rs[i]) for i in range(self.d1)]
        cs = [str(cs[i]) for i in range(self.d2)]
        times = pd.DataFrame(self.times,index=rs,columns=cs)
        fig = sns.heatmap(times, annot=True)
        fig.set(xlabel=self.labels[0], ylabel=self.labels[1])

    def err_heat(self,rs,cs):
        rs = [str(rs[i]) for i in range(self.d1)]
        cs = [str(cs[i]) for i in range(self.d2)]
        errs = pd.DataFrame(self.errs,index=rs,columns=cs)
        fig = sns.heatmap(errs, annot=True)
        fig.set(xlabel=self.labels[0], ylabel=self.labels[1])
        
    def acc_heat(self,rs,cs):
        rs = [str(rs[i]) for i in range(self.d1)]
        cs = [str(cs[i]) for i in range(self.d2)]
        accs = pd.DataFrame(self.accs,index=rs,columns=cs)
        fig = sns.heatmap(accs, annot=True)
        fig.set(xlabel=self.labels[0], ylabel=self.labels[1])
    
    def save_object(self,filename):
        the_big_tensor = numpy.zeros([self.d1,self.d2,3,self.sims[0][0].num_epochs)
        for k in range(self.d1):
            for j in range(self.d2):
                the_big_tensor[k,j,0,:] = self.sims[k][j].times
                the_big_tensor[k,j,1,:] = self.sims[k][j].errv
                the_big_tensor[k,j,2,:] = self.sims[k][j].accv
        np.save('/home/james/Models/'+filename,the_big_tensor)
                

class DataLoader(object):
    def __init__(self,filename,labels):
        self.the_big_tensor = np.load('/home/james/Models/'+filename)
        for k in range(self.d1):
            for j in range(self.d2):
                self.times[k][j],self.errs[k][j],self.accs[k][j] = self.the_big_tensor[k,j,0,:],self.the_big_tensor[k,j,1,:],self.the_big_tensor[k,j,2,:]
        
               