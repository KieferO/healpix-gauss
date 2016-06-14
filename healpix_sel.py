#!/usr/bin/env python

from __future__ import print_function

import sys

from healpy.pixelfunc import *
import healpy
import numpy as np
import matplotlib.pyplot as plt

def pix_circle(nside, ipix, radius, nest=False):
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

def main():
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

if __name__ == '__main__':
    sys.exit(main())
