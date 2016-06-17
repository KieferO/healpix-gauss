#!/usr/bin/env python

from __future__ import print_function

import sys
import math
import random
import cPickle as pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from healpy.pixelfunc import *
import healpy
import numpy

import scipy
import scipy.misc
import scipy.sparse

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
        return math.exp(- (50 * dist) ** 2)
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

def scale_radius_trial(points, kernel, neighborsof, tol):
    i = 0
    center = random.sample(points, 1)[0]
    far = center
    while True:
        i += 1
        m_val = 100
        m_nei = None
        for n in neighborsof(far):
            val = kernel(center, n)
            if val < m_val:
                m_val = val
                m_nei = n
        if m_val < tol:
            return i
        far = m_nei

def find_scale_radius(*args):
    trials = 10
    return int(sum([ scale_radius_trial(*args) for _ in range(trials) ]) /
               float(trials))

def pts2cov(points, kernel, sparse=True):
    if sparse:
        cov = scipy.sparse.dok_matrix((len(points),) * 2, numpy.float64)
    else:
        cov = numpy.zeros((len(points),) * 2)
    for i, pt1 in enumerate(points):
        for j, pt2 in enumerate(points[i:]):
            variance = kernel(pt1, pt2)
            if abs(variance) > 1e-6:
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
    hotspot_r = 16
    margin_r = find_scale_radius(points, kernel, neighborsof, 1e-6)
    gutter_r = find_scale_radius(points, kernel, neighborsof, 1e-12)
    while points:
        center = 138 #random.sample(points, 1)[0]
        region = circle(center, neighborsof, hotspot_r + gutter_r)
        margin = circle(center, neighborsof, hotspot_r + margin_r)
        hotspot = circle(center, neighborsof, hotspot_r)
        useful_points = len(points & hotspot)
        if hotspot_r > 2 and useful_points * 2 < len(hotspot):
            hotspot_r -= 1
        print(len(region), hotspot_r, useful_points, len(points))
        points -= hotspot
        region = list(region)
        region.sort()
        cov = pts2cov(region, kernel, sparse=False)
        local_invcov = numpy.linalg.inv(cov)
        scipy.misc.imsave('local_invcov.png', local_invcov)
        scipy.misc.imsave('local_cov.png', cov)
        scipy.misc.imsave('local_check.png', numpy.dot(local_invcov, cov))
        for i, pt1 in enumerate(region):
            for j, pt2 in enumerate(region[i:]):
                if set((pt1, pt2)) < margin and set((pt1, pt2)) & hotspot:
                    val = local_invcov[i][j + i]
                    #if val and math.log10(abs(val)) > -6:
                    if True:
                        #global_invcov[frozenset((pt1, pt2))] =
                        if not global_invcov[pt1, pt2]:
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
    pix_row = vec2pix(order2nside(order),
                      0.22203510307746671,
                      -0.33229901477976281,
                      0.91666666666666663)
    row2fig('invcovrow.png', invcov[pix_row], nest=NEST)
    #scipy.misc.imsave('invcov.png', invcov)
    with open('invcovrow.txt', 'w') as f:
        for val in invcov[pix_row]:
            print(val, file=f)
    return -1
    cov = pts2cov(pts, kern, sparse=False)
    scipy.misc.imsave('cov.png', cov)
    check_mat = numpy.dot(invcov, cov)
    scipy.misc.imsave('check.png', check_mat)
    row2fig('covrow.png', cov[pix_row], nest=NEST)
    row2fig('checkrow.png', check_mat[pix_row], nest=NEST)

def run_checks():
    scipy.misc.imsave('cov.png', cov)
    scipy.misc.imsave('check.png', check_mat)
    scipy.misc.imsave('slow_invcov.png', numpy.linalg.inv(cov))
    scipy.misc.imsave('invcov.png', invcov)


if __name__ == '__main__':
    sys.exit(main())
