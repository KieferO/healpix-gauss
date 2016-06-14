#!/usr/bin/env python

from __future__ import print_function

import sys
import math
import random

from healpy.pixelfunc import *
import healpy
import numpy as np
import matplotlib.pyplot as plt

import scipy
import scipy.misc

def pix_circle(nside, ipix, radius, nest=False):
    assert False
    unvisited = set([ipix])
    visited = set()
    for _ in range(radius):
        while unvisited:
            ipix = unvisited.pop()
            neighbours = set(get_all_neighbours(nside, ipix, phi=None,
                                                nest=nest))
            neighbours -= visited
            neighbours.discard(-1)
            visited |= neighbours
            visited.add(ipix)
        unvisited |= visited
    return visited

def draw_circles():
    assert False
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    nside = 64
    npix = nside2npix(nside)
    xmap = np.zeros(npix)
    for i in range(16, 0, -1):
        highlight = pix_circle(nside, 0, i, nest=False)
        for h in highlight:
            xmap[h] = i
    img_x = healpy.visufunc.mollview(
        xmap,
        nest=False,
        return_projected_map=True)
    ax.imshow(img_x)
    ax.axis('off')
    fig.savefig('indicator.png')
    return 0

# Read in a list of points.  Remember their indicies.  You'll need them later.
# Accept a distance kernel.  It should return the variance between two
# points.
# Accept neighborsof: given a point, return the neighbors of this point.
# def pts2invcov(points, kernel, neighborsof):
# Generator: points
# function(point, point) -> float: kernel
# function(point) -> [point, ...]: neighborsof

def cast_neighborsof(order, nest=False):
    nside = order2nside(order)
    def neighborsof(ipix):
        outarr = get_all_neighbours(nside, ipix, phi=None, nest=nest)
        return [ x for x in outarr if x != -1 ]
    return neighborsof

def pointsof(order):
    return range(nside2npix(order2nside(order)))

def cast_kernel(order, nest=False):
    nside = order2nside(order)
    def kernel(pt1, pt2):
        assert nest
        dist = 1.0 - np.dot(pix2vec(nside, pt1, nest=nest),
                            pix2vec(nside, pt2, nest=nest))
        dist *= math.pi
        return math.exp(-dist * 30)
    return kernel

def circle(ipix, neighborsof, radius):
    unvisited = set([ipix])
    visited = set()
    for _ in range(radius):
        while unvisited:
            ipix = unvisited.pop()
            neighbours = set(neighborsof(ipix))
            neighbours -= visited
            visited |= neighbours
            visited.add(ipix)
        unvisited |= visited
    return visited

def pts2cov(points, kernel):
    cov = np.zeros((len(points),) * 2)
    for i, pt1 in enumerate(points):
        for j, pt2 in enumerate(points[i:]):
            variance = kernel(pt1, pt2)
            cov[i][i + j] = variance
            cov[j + i][i] = variance
        cov[i][i] = kernel(pt1, pt1)
    return cov

def pts2invcov(points, kernel, neighborsof):
    random.shuffle(points)
    points = set(points)
    global_invcov = dict()
    while points:
        center = points.pop()
        region = circle(center, neighborsof, 12)
        hotspot = circle(center, neighborsof, 10)
        points -= hotspot
        region = list(region)
        cov = pts2cov(region, kernel)
        local_invcov = np.linalg.inv(cov)
        for i, pt1 in enumerate(region):
            for j, pt2 in enumerate(region[i:]):
                if set((pt1, pt2)) <= hotspot:
                    global_invcov[frozenset((pt1, pt2))] = local_invcov[i][j + i]
    return global_invcov

def dict2mat(invcov, order):
    size = nside2npix(order2nside(order))
    mat = np.zeros((size, size))
    for ent in invcov.keys():
        if len(ent) > 1:
            i, j = ent
        else:
            i, j = list(ent) * 2
        mat[i][j] = invcov[ent]
        mat[j][i] = invcov[ent]
    return mat

def main():
    order = 4
    d = pts2invcov(
        pointsof(order),
        cast_kernel(order, nest=True),
        cast_neighborsof(order, nest=True))
    return -1
    invcov = dict2mat(d, order)
    cov = pts2cov(pointsof(order), cast_kernel(order, nest=True))
    check_mat = np.dot(cov, invcov)
    scipy.misc.imsave('cov.png', cov)
    scipy.misc.imsave('check.png', check_mat)
    scipy.misc.imsave('slow_invcov.png', np.linalg.inv(cov))
    scipy.misc.imsave('invcov.png', invcov)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    img_x = healpy.visufunc.mollview(
        invcov[113],
        nest=True,
        return_projected_map=True)
    ax.imshow(img_x)
    ax.axis('off')
    fig.savefig('randrow.png')
    return 0

if __name__ == '__main__':
    sys.exit(main())
