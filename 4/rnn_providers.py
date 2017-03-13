import os
import numpy as np
seed = 123
rng = np.random.RandomState(seed)
from mlp.data_providers import MSD10GenreDataProvider, MSD25GenreDataProvider

class EnsembleMSD10DataProvider(MSD10GenreDataProvider):
    """Data provider for MNIST dataset which transforms images using a variety of possible autoencoders (potentially) trained with added noise"""

    def __init__(self, which_set='train', batch_size=50, max_num_batches=-1,
                 shuffle_order=True, rng=rng,div=2):
        """Create a new augmented MNIST data provider object.

        Args:
            which_set: One of 'train', 'valid' or 'test'. Determines which
                portion of the MNIST data this object should provide.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
            transformer: Function which takes an `inputs` array of shape
                (batch_size, input_dim) corresponding to a batch of input
                images and a `rng` random number generator object (i.e. a
                call signature `transformer(inputs, rng)`) and applies a
                potentiall random set of transformations to some / all of the
                input images as each new batch is returned when iterating over
                the data provider.
        """
        super(GapMSD10DataProvider,self).__init__(
            which_set, batch_size, max_num_batches, shuffle_order, rng)


    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        inputs_batch, targets_batch = super(GapMSD10DataProvider,self).next()
        return inputs_batch, targets_batch

class GapMSD25DataProvider(MSD25GenreDataProvider):
    """Data provider for MNIST dataset which transforms images using a variety of possible autoencoders (potentially) trained with added noise"""

    def __init__(self, which_set='train', batch_size=50, max_num_batches=-1,
                 shuffle_order=True, rng=rng,gap=range(10)):
        """Create a new augmented MNIST data provider object.

        Args:
            which_set: One of 'train', 'valid' or 'test'. Determines which
                portion of the MNIST data this object should provide.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
            transformer: Function which takes an `inputs` array of shape
                (batch_size, input_dim) corresponding to a batch of input
                images and a `rng` random number generator object (i.e. a
                call signature `transformer(inputs, rng)`) and applies a
                potentiall random set of transformations to some / all of the
                input images as each new batch is returned when iterating over
                the data provider.
        """
        super(GapMSD25DataProvider,self).__init__(
            which_set, batch_size, max_num_batches, shuffle_order, rng)
        self.gap = gap
        self.inputs = self.inputs.reshape(-1,120,25)
        targets2 = self.inputs[:,self.gap,:].reshape(self.inputs.shape[0],-1)
        print targets2.shape
        self.targets = np.transpose(np.array([self.targets]))
        print self.targets.shape
        self.targets = np.concatenate([self.targets,targets2],1)
        self.inputs = np.delete(self.inputs,self.gap,1)
        self.inputs.reshape(self.inputs.shape[0],-1)
        
    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        inputs_batch, targets_batch = super(GapMSD25DataProvider,self).next()
        return inputs_batch, targets_batch[:,0] , targets_batch[:,1:]