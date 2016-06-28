#!/usr/bin/env python

from __future__ import print_function

import numpy
import scipy.misc

import sys

import cart

def dump_mat(mat, outf):
    nrows = len(mat)
    ncols = len(mat[0])
    fl = str(max([ len(str(n)) for n in (nrows, ncols) ]))
    ifmt = '{i: >'+fl+'},{j: >'+fl+'}'
    valfmt = '{val:.20e}'
    for i in range(nrows):
        for j in range(ncols):
            val = '{: <27s}'.format(valfmt.format(val=mat[i][j]))
            outf.write(ifmt.format(i=i, j=j) + ',' + val + '\n')

def main():
    order = int(sys.argv[1])
    cov, inv_cov, inv_cov_est, t_direct, t_patches = cart.get_mats(order)
    with open('cov-%d.csv' % order, 'w') as outf:
        dump_mat(cov, outf)

if __name__ == '__main__':
    sys.exit(main())
