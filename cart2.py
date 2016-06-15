#!/usr/bin/env python

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def make_toroidal_metric(shape):
    n_dim = len(shape)

    def metric(x, y):
        """
        Returns distance squared between two sets of points in a periodic
        Cartesian space.

        Parameters
        ----------
        x : np.ndarray
            The first set of coordinates.
            x.shape = (# of points, # of dimensions)
        y : np.ndarray
            The second set of coordinates.
            x.shape = (# of points, # of dimensions)

        Returns
        -------
        np.ndarray
            The distance between each pair of points in `x` and `y`.
        """
        try:
            assert x.shape[1] == n_dim
        except:
            raise ValueError('`x` must have shape (# of points, # of dimensions).')

        try:
            assert x.shape == y.shape
        except:
            raise ValueError('Shapes of `x` and `y` must match.')

        return np.sum(
            np.minimum(np.mod(x-y, shape), np.mod(y-x, shape))**2,
            axis=1
        )

    return metric

def main():
    shape = np.array([10])
    n_points = 10
    n_dim = len(shape)

    metric = make_toroidal_metric(shape)

    x = shape[None,:] * np.random.random(size=(n_points, n_dim))
    y = shape[None,:] * np.random.random(size=(n_points, n_dim))

    return 0

if __name__ == '__main__':
    main()
