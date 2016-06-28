#!/usr/bin/env python

from __future__ import print_function

import sys

import numpy
import scipy.misc

def main():
    size = int(sys.argv[1])
    infn = sys.argv[2]
    outmat = numpy.zeros((size, size))
    with open(infn) as infile:
        for line in infile:
            i, j, val = [ x.strip() for x in line.strip().split(',') ]
            i, j = int(i), int(j)
            val = float(val)
            outmat[i][j] = val
    scipy.misc.imsave(infn.replace('.csv', '.png'), outmat)
    return 0

if __name__ == '__main__':
    sys.exit(main())
