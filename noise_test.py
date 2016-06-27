#!/usr/bin/env python

from __future__ import print_function

import numpy
import scipy.misc

import sys

import cart

def rand_symm(*args):
    mat = numpy.random.rand(*args)
    return (mat + numpy.transpose(mat)) / 2.0

def main():
    matsizes = 75
    noises = 75
    #cov, inv_cov, inv_cov_est, t_direct, t_patches = cart.get_mats(4)
    #matsize = len(cov)
    matsize = 12288
    print(matsize)
    mat = rand_symm(matsize, matsize)
    invmat = numpy.linalg.inv(mat)
    outfile = open('randsymm_error.csv', 'a')
    for j in range(noises):
        print(j)
        noise = 10 ** (float(j) * 5 / noises - 5.2)
        #mat = cov
        noise_mat = rand_symm(matsize, matsize) * noise
        #invmat = inv_cov
        check_mat = numpy.dot(invmat, mat + noise_mat)
        check_mat -= numpy.identity(matsize)
        print(matsize, noise, numpy.median(abs(check_mat)),
              sep=',', file=outfile)
    return 0

if __name__ == '__main__':
    sys.exit(main())
