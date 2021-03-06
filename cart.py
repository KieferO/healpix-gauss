#!/usr/bin/env python

from __future__ import print_function

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import colors

from healpy.pixelfunc import *
import healpy

import math
import random
import sys

def row2fig(filename, row, nest=False):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    img_x = healpy.visufunc.mollview(
        row,
        nest=nest,
        return_projected_map=True)
    ax.imshow(img_x)
    ax.axis('off')
    fig.savefig(filename)

def make_toroidal_metric(shape, l=2, manhattan=False):
    n_dim = len(shape)
    shape = np.array(shape)
    f_reduce = np.max if manhattan else np.sum

    def metric(x, y):
        """
        Returns distance squared between two sets of points in a periodic
        Cartesian space.

        Parameters
        ----------
        x : np.ndarray
            The first set of coordinates.
            x.shape = (# of dimensions, # of points)
        y : np.ndarray
            The second set of coordinates.
            y.shape = (# of dimensions, # of points)

        Returns
        -------
        np.ndarray
            The squared distance between each pair of points in `x` and `y`.
        """
        if x.shape[0] != n_dim:
            msg = '{} != {}: '
            msg.format(x.shape[0], n_dim)
            msg += '`x` must have shape (# of dimensions, # of points).'
            raise ValueError(msg)

        if x.shape != y.shape:
            msg = '{} != {}: '.format(x.shape, y.shape)
            msg += 'Shapes of `x` and `y` must match.'
            raise ValueError(msg)

        out_vec = f_reduce(
            np.abs(np.minimum(
                np.remainder(x-y, shape[:,None]),
                np.remainder(y-x, shape[:,None])
            ))**l,
            axis=0
        )**(1./l)
        return out_vec

    return metric

def make_healpix_metric(order, nest=False):
    def metric(x, y):
        '''
        Returns the distance in radians between two healpix pixels.

        Parameters
        ----------
        x : numpy.ndarray
            The first set of coordinates.
            x.shape = (# of points,)
        y : numpy.ndarray
            The second set of coordinates.
            y.shape = (# of points,)

        Returns
        -------
        numpy.ndarray
            The squared distance between each pair of points in `x` and `y`.
        '''
        if hasattr(x, 'shape') or hasattr(y, 'shape'):
            if x.shape != y.shape:
                msg = '{} != {}: '.format(x.shape, y.shape)
                msg += 'Shapes of `x` and `y` must match.'
                raise ValueError(msg)
        else:
            if hasattr(x, '__getitem__'):
                x = np.array(x)
                y = np.array(y)
            else:
                x = np.array((x,))
                y = np.array((y,))

        shape = x.shape
        x.shape = (x.size,)
        y.shape = (y.size,)
        nside = order2nside(order)
        def our_pix2vec(ipix):
            outvec = np.array(pix2vec(nside, ipix, nest=nest))
            return outvec
        d = 1.0 - np.einsum('ij,ij->j', our_pix2vec(x), our_pix2vec(y))
        d *= math.pi
        d.shape = shape
        if d.shape == (1,):
            d = d[0]
        return d
    return metric


def flattened_grid(shape):
    grid = np.indices(shape)
    grid.shape = (grid.shape[0], np.prod(grid.shape[1:]))
    return grid

def dist_matrix(points, metric):
    shape = points.shape
    if len(shape) == 1:
        shape = (1, shape[0])
    idx = np.indices((shape[1], shape[1]))
    idx.shape = (2, shape[1]**2)
    dist = metric(points[...,idx[0]], points[...,idx[1]])
    dist.shape = (shape[1], shape[1])
    return dist

def make_exp_kernel(scale_length):
    def kernel(dist):
        return np.exp(-dist/scale_length)
    return kernel

class CartesianPatchIterator(object):
    def __init__(self, points, patch_centers, metric, r_core, r_border):
        self._points = points
        self._patch_centers = patch_centers
        self._metric = metric
        self._r_core = r_core
        self._r_border = r_border

        self._n_patches = patch_centers.shape[-1]
        if len(points.shape) > 1:
            self._n_dim, self._n_points = points.shape
        else:
            self._n_dim, self._n_points = (1,) + points.shape
        self._x0 = np.empty((self._n_dim, self._n_points), dtype='f8')

        self._current = 0

    def __iter__(self):
        return self

    def next(self):
        """
        Return a boolean array identifying the core of the next patch,
        and a second boolean array identifying the core+border.
        """
        if self._current >= self._n_patches:
            raise StopIteration
        self._x0[:,:] = self._patch_centers[:,self._current][:,None]
        center_dist = self._metric(self._points, self._x0)
        self._current += 1
        return ((center_dist <= self._r_core),
                (center_dist <= self._r_border))

class HealpixPatchIterator(object):
    def __init__(self, points, patch_centers, metric, r_core, r_border):
        self._points = points
        self._patch_centers = patch_centers
        self._metric = metric
        self._r_core = r_core
        self._r_border = r_border

        self._n_patches = patch_centers.shape[-1]
        self._n_points = points.shape[-1]
        self._x0 = np.zeros((self._n_points,), dtype=int)

        self._current = 0

    def __iter__(self):
        return self

    def next(self):
        """
        Return a boolean array identifying the core of the next patch,
        and a second boolean array identifying the core+border.
        """
        if self._current >= self._n_patches:
            raise StopIteration
        self._x0 = np.ones(self._n_points, dtype=int) * self._patch_centers[self._current]
        center_dist = self._metric(self._points, self._x0)
        self._current += 1
        return ((center_dist <= self._r_core),
                (center_dist <= self._r_border))

def invert_by_patches(points, metric, kernel, patch_iter):
    # n_patches = patch_centers.shape[1]
    # n_points = points.shape[1]

    # Determine distance of every point to every patch center
    # patch_dist = np.empty((n_patches, n_points), dtype='f8')
    # x0 = np.empty(points.shape)
    #
    # for k in range(n_patches):
    #     x0[:,:] = patch_centers[:,k][:,None]
    #     patch_dist[k,:] = metric(x0, points)
    #
    # # Find nearest patch for each point
    # nearest_patch_idx = np.argmin(patch_dist, axis=0)

    # for patch_idx in range(n_patches):
    #     point_idx = (nearest_patch_idx == patch_idx)
    #     cov = kernel(dist_matrix(points[:,point_idx], metric))
    #     inv_cov = np.linalg.inv(cov)

    nearest_patch_idx = np.zeros(points.shape[-1], dtype='f8')

    n_points = points.shape[-1]
    inv_cov_full = np.zeros((n_points, n_points), dtype='f8')

    for k, (idx_core, idx_border) in enumerate(patch_iter):
        cov_border = kernel(dist_matrix(points[...,idx_border], metric))
        inv_cov_border = np.linalg.inv(cov_border)
        # print(inv_cov_border[:4,:4])

        # Extract the inverse covariance for the core
        idx_core_in_border = idx_core[idx_border]
        inv_cov_core = inv_cov_border[idx_core_in_border,:][:,idx_core_in_border]
        # print(inv_cov_core)

        core_rows = np.where(idx_core)[0]
        # print(core_rows)
        core_rows, core_cols = np.meshgrid(core_rows, core_rows)
        # print(core_rows.shape, core_cols.shape, inv_cov_core.shape)

        core_rows.shape = (core_rows.size,)
        core_cols.shape = (core_cols.size,)
        # inv_cov_core.shape = (inv_cov_core.size,)

        # print(inv_cov_core[:4,:4])
        # print(core_rows)
        # print(core_cols)
        inv_cov_full[core_rows,core_cols] = inv_cov_core.flat

        nearest_patch_idx[idx_border] += 1.

    return inv_cov_full, nearest_patch_idx

def get_patch_centers(order):
    s_order = order - 2
    c0s = np.array(range(nside2npix(order2nside(s_order))))
    return vec2pix(order2nside(order), *pix2vec(order2nside(s_order), c0s))

def get_mats(order):
    from time import time
    npix = nside2npix(order2nside(order))
    shape = np.array([npix, npix])
    patch_centers = get_patch_centers(order)
    r_core = 0.1
    r_border = 0.2

    n_dim = len(shape)

    metric = make_healpix_metric(order)
    inv_scale_length = 60.0
    scale_length = 1.0 / inv_scale_length
    kernel = make_exp_kernel(scale_length)

    grid = np.array(range(npix))
    x0 = np.ones(npix, dtype=int) * 365

    dist = metric(grid, x0)
    dist_mat = dist_matrix(grid, metric)

    t0 = time()

    cov = kernel(dist_mat)
    inv_cov = np.linalg.inv(cov)

    t1 = time()

    patch_metric = metric
    t1 = time()
    patch_iter = HealpixPatchIterator(
        grid,
        patch_centers,
        patch_metric,
        r_core,
        r_border
    )
    inv_cov_est, patch_identity = invert_by_patches(
        grid,
        metric,
        kernel,
        patch_iter
    )
    return cov, inv_cov, inv_cov_est, t1 - t0, time() - t1

def main():
    order = 4
    cov, inv_cov, inv_cov_est, t_direct, t_patches = get_mats(order)
    header = '=== inv_scale_length:{:.1f}: ==='
    header = header.format(inv_scale_length)
    print(header, file=logf)
    print('Time to invert directly:   {:.4f} s'.format(t_direct), file=logf)
    maybe_zero = inv_cov_est - inv_cov_est.T
    for i in range(npix):
        for j in range(npix):
            print(maybe_zero[i][j])
    return 0
    #patch_identity.shape = shape

    t2 = time()
    print('Time to invert by patches: {:.4f} s'.format(t2-t1), file=logf)
    I = np.dot(inv_cov, cov)
    I_est = np.dot(inv_cov_est, cov)

    t3 = time()
    print('Time to calculate checks:  {:.4f} s'.format(t3-t2), file=logf)

    I_diag = I[np.diag_indices(I.shape[0])]
    I_diag_dev = np.max(np.abs(I_diag-1.))
    I[np.diag_indices(I.shape[0])] = 0.
    I_off_diag_dev = np.max(np.abs(I))

    print('C^-1 C:', file=logf)
    print('  * max. abs. dev. along diagonal: {:.3g}'.format(I_diag_dev), file=logf)
    print('  * max. abs. dev. off diagonal: {:.3g}'.format(I_off_diag_dev), file=logf)

    I_diag = I_est[np.diag_indices(I_est.shape[0])]
    I_diag_dev = np.max(np.abs(I_diag-1.))
    I_est[np.diag_indices(I_est.shape[0])] = 0.
    I_off_diag_dev = np.max(np.abs(I_est))

    print('(C^-1)_{est} C:', file=logf)
    print('  * max. abs. dev. along diagonal: {:.3g}'.format(I_diag_dev), file=logf)
    print('  * max. abs. dev. off diagonal: {:.3g}'.format(I_off_diag_dev), file=logf)

    #plt.imsave('dist.png', dist, cmap=colors.inferno_r)
    row2fig('dist.png', dist)
    plt.imsave('dist_matrix.png', dist_mat, cmap=colors.inferno_r)
    plt.imsave('cov.png', cov, cmap=colors.inferno_r)
    vmax = np.max(np.abs(inv_cov))
    plt.imsave('inv_cov.png', inv_cov, cmap='coolwarm_r', vmin=-vmax, vmax=vmax)
    vmax = np.max(np.abs(inv_cov_est))
    plt.imsave('inv_cov_est.png', inv_cov_est, cmap='coolwarm_r', vmin=-vmax, vmax=vmax)
    row2fig('inv_cov_est_row.png', inv_cov_est[1387])
    with open('inv_cov_est_diag.txt', 'w') as icer:
        for i in range(npix):
            print(np.max(inv_cov_est[i]), file=icer)
    with open('inv_cov_diag.txt', 'w') as icer:
        for i in range(npix):
            print(np.max(inv_cov[i]), file=icer)
    #plt.imsave('patches.png', patch_identity, cmap=colors.inferno_r)
    row2fig('patches.png', patch_identity)

    return 0

if __name__ == '__main__':
    main()
