#!/usr/bin/env python

from __future__ import print_function

import sys
import math
import random
import cPickle as pickle
import time

from healpy.pixelfunc import *
import healpy
import numpy
import matplotlib.pyplot as plt

import scipy
import scipy.misc
import scipy.sparse

import cProfile as profile

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
    xmap = numpy.zeros(npix)
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
        dist = 1.0 - numpy.dot(pix2vec(nside, pt1, nest=nest),
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

def pts2cov(points, kernel, sparse=True):
    if sparse:
        cov = scipy.sparse.dok_matrix((len(points),) * 2, numpy.float64)
    else:
        cov = numpy.zeros((len(points),) * 2)
    for i, pt1 in enumerate(points):
        for j, pt2 in enumerate(points[i:]):
            variance = kernel(pt1, pt2)
            if math.log10(abs(variance)) > -6:
                cov[i, i + j] = variance
                cov[j + i, i] = variance
        cov[i, i] = kernel(pt1, pt1)
    if sparse:
        cov = cov.tocsc()
    return cov

def pts2invcov(points, kernel, neighborsof):
    points = set(points)
    size = len(points)
    global_invcov = scipy.sparse.dok_matrix((size, size), numpy.float64)
    #global_invcov = dict()
    while points:
        center = random.sample(points, 1)[0]
        region = circle(center, neighborsof, 15)
        hotspot = circle(center, neighborsof, 8)
        print(len(region), len(hotspot))
        points -= hotspot
        region = list(region)
        cov = pts2cov(region, kernel, sparse=False)
        local_invcov = numpy.linalg.inv(cov)
        for i, pt1 in enumerate(region):
            for j, pt2 in enumerate(region[i:]):
                if set((pt1, pt2)) & hotspot:
                    val = local_invcov[i][j + i]
                    if val and math.log10(abs(val)) > -6:
                        #global_invcov[frozenset((pt1, pt2))] =
                        global_invcov[pt1, pt2] = val
                        global_invcov[pt2, pt1] = val
    global_invcov = global_invcov.tocsr()
    return global_invcov

def dict2mat(invcov, order):
    size = nside2npix(order2nside(order))
    mat = numpy.zeros((size, size))
    for ent in invcov.keys():
        if len(ent) > 1:
            i, j = ent
        else:
            i, j = list(ent) * 2
        mat[i][j] = invcov[ent]
        mat[j][i] = invcov[ent]
    return mat

def ss_matmul(csr, csc, size):
    outmat = numpy.zeros((size,) * 2)
    for i in range(size):
        for j in range(size):
            row = csr.getrow(i).todense()
            col = csc.getcol(j).todense()
            outmat[i, j] = numpy.dot(row, col)
        print(i)
    return outmat

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

def main():
    order = 4
    pts = pointsof(order)
    NEST = False
    kern = cast_kernel(order, nest=NEST)
    neighbor_func = cast_neighborsof(order, nest=NEST)
    invcov = pts2invcov(pts, kern, neighbor_func)
    invcov = numpy.array(invcov.todense())
    cov = pts2cov(pts, kern, sparse=False)
    check_mat = numpy.dot(invcov, cov)
    scipy.misc.imsave('invcov.png', invcov)
    scipy.misc.imsave('cov.png', cov)
    scipy.misc.imsave('check.png', check_mat)
    row2fig('covrow.png', cov[37], nest=NEST)
    row2fig('invcovrow.png', invcov[37], nest=NEST)

def profilerun():
    order = 3
    pts = pointsof(order)
    NEST = False
    kern = cast_kernel(order, nest=NEST)
    neighbor_func = cast_neighborsof(order, nest=NEST)
    return pts2invcov(pts, kern, neighbor_func)

def prof():
    profile.run('profilerun()')

def run_checks():
    scipy.misc.imsave('cov.png', cov)
    scipy.misc.imsave('check.png', check_mat)
    scipy.misc.imsave('slow_invcov.png', numpy.linalg.inv(cov))
    scipy.misc.imsave('invcov.png', invcov)


if __name__ == '__main__':
    sys.exit(prof())
