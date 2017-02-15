import os
import numpy as np
seed = 123
rng = np.random.RandomState(seed)
from mlp.data_providers import MSD10GenreDataProvider, MSD25GenreDataProvider

class AugmentedMSD10DataProvider(MSD10GenreDataProvider):
    """Data provider for MNIST dataset which transforms images using a variety of possible autoencoders (potentially) trained with added noise"""

    def __init__(self, which_set='train', batch_size=50, max_num_batches=-1,
                 shuffle_order=True, rng=rng,fraction=0.25,std=0.05):
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
        super().__init__(
            which_set, batch_size, max_num_batches, shuffle_order, rng)
        self.fraction = fraction
        self.std = std
        
        inds = np.random.choice(range(self.targets.shape[0]),int(round(self.fraction*(self.targets.shape[0]))))
        self.inputs = np.concatenate([self.inputs,[self.inputs[i,:] for i in inds]])
        self.targets = np.concatenate([self.targets,[self.targets[i] for i in inds]])
        self._update_num_batches()
        self.shuffle_order = shuffle_order
        self._current_order = np.arange(self.inputs.shape[0])
    
    def transform(self,inputs_batch):
        if np.random.choice(2,size=(1,1),p=(self.fraction,1-self.fraction))[0][0]:
            return inputs_batch + np.random.normal(0, self.std, inputs_batch.shape)
        else:
            return inputs_batch
        

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        inputs_batch, targets_batch = super().next()
        transformed_inputs_batch = self.transform(inputs_batch)
        return transformed_inputs_batch, targets_batch

class AugmentedMSD25DataProvider(MSD10GenreDataProvider):
    """Data provider for MNIST dataset which transforms images using a variety of possible autoencoders (potentially) trained with added noise"""

    def __init__(self, which_set='train', batch_size=50, max_num_batches=-1,
                 shuffle_order=True, rng=rng,fraction=0.25,std=0.05):
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
        super().__init__(
            which_set, batch_size, max_num_batches, shuffle_order, rng)
        self.fraction = fraction
        self.std = std
        
        inds = np.random.choice(range(self.targets.shape[0]),int(round(self.fraction*(self.targets.shape[0]))))
        self.inputs = np.concatenate([self.inputs,[self.inputs[i,:] for i in inds]])
        self.targets = np.concatenate([self.targets,[self.targets[i] for i in inds]])
        self._update_num_batches()
        self.shuffle_order = shuffle_order
        self._current_order = np.arange(self.inputs.shape[0])
    
    def transform(self,inputs_batch):
        if np.random.choice(2,size=(1,1),p=(self.fraction,1-self.fraction))[0][0]:
            return inputs_batch + np.random.normal(0, self.std, inputs_batch.shape)
        else:
            return inputs_batch
        

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        inputs_batch, targets_batch = super().next()
        transformed_inputs_batch = self.transform(inputs_batch)
        return transformed_inputs_batch, targets_batch