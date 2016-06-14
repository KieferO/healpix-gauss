#!/usr/bin/env python

from __future__ import print_function

import sys
import math
import time

import numpy
from numpy import linalg

import scipy
import scipy.misc

def mat2d(size):
    field = []
    for x in range(size):
        for y in range(size):
            field.append((float(x), float(y)))
    return numpy.reshape(field, (size, size, 2))

def mat2cov(field):
    covsz = field.shape[0] ** 2
    cov = numpy.zeros((covsz, covsz))
    for i in range(covsz):
        src_x, src_y = field[i / field.shape[0]][i % field.shape[0]]
        for j in range(covsz):
            tgt_x, tgt_y = field[j / field.shape[0]][j % field.shape[0]]
            dist = math.sqrt((src_x - tgt_x) ** 2 +
                             (src_y - tgt_y) ** 2)
            cov[i][j] = math.exp(-dist)
    return cov

def cov2row(cov, x, y):
    SZ = int(math.sqrt(cov.shape[0]))
    row = x + SZ * y
    return cov[row].reshape((SZ, SZ))

def get_margins(field, size, margin, s_i):
    margin_sinister = margin
    margin_dexter = margin
    idx_from = s_i * size
    idx_to = (s_i + 1) * size
    if idx_from - margin_sinister < 0:
        adj = (idx_from - margin_sinister)
        margin_dexter -= adj
        margin_sinister += adj
    if idx_to + margin_dexter > len(field):
        adj = idx_to + margin_dexter - len(field)
        margin_sinister += adj
        margin_dexter -= adj
    idx_from -= margin_sinister
    idx_to += margin_dexter
    return (idx_from, idx_to, margin_sinister, margin_dexter)

def write_to_invcov(field, size, margin, n_superblocks, out_invcov, offsets):
    i_offset, j_offset = offsets
    for s_i in range(n_superblocks - i_offset):
        rlmargins = get_margins(field, size, margin, s_i)
        row_from, row_to, margin_left, margin_right = rlmargins
        if i_offset:
            row_from += size / 2
            row_to += size / 2
        for s_j in range(n_superblocks - j_offset):
            tbmargins = get_margins(field, size, margin, s_j)
            col_from, col_to, margin_top, margin_bot = tbmargins
            if j_offset:
                col_from += size / 2
                col_to += size / 2
            sub_mat = field[row_from:row_to,col_from:col_to]
            cov = mat2cov(sub_mat)
            invcov = linalg.inv(cov)
            for r_i in range(size):
                for r_j in range(size):
                    invcov_row = (margin_top + r_i) * (size + margin * 2)
                    invcov_row += margin_left + r_j
                    r_orig = ((size / 2) * i_offset + s_i * size + r_i, (size / 2) * j_offset + s_j * size + r_j)
                    out_row = r_orig[0] * len(field) + r_orig[1]
                    for c_i in range(size):
                        for c_j in range(size):
                            invcov_col = (margin_top + c_i) * (size + margin * 2)
                            invcov_col += margin_left + c_j
                            val = invcov[invcov_row][invcov_col]
                            c_orig = ((size / 2) * i_offset + s_i * size + c_i, (size / 2) * j_offset + s_j * size + c_j)
                            out_col = c_orig[0] * len(field) + c_orig[1]
                            out_invcov[out_row][out_col] = 4 * j_offset + 2 * i_offset

def quick_inv(field, size, margin):
    out_invcov = numpy.zeros((len(field) ** 2,) * 2)
    # Tile the field with rectangles size x size.
    n_superblocks = (len(field) / size) + (1 if (len(field) % size) else 0)
    for i_offset in range(2):
        for j_offset in range(2):
            offsets = (i_offset, j_offset)
            write_to_invcov(field, size, margin, n_superblocks, out_invcov,
                            offsets)
    return out_invcov

def main():
    SZ = 32
    cov = mat2cov(mat2d(SZ))
    field = mat2d(SZ)
    cov = mat2cov(field)
    invcov = quick_inv(field, 8, 0)
    slow_invcov = numpy.linalg.inv(cov)
    scipy.misc.imsave('invcov.png', invcov)
    scipy.misc.imsave('slow-invcov.png', slow_invcov)
    check = numpy.dot(cov, invcov)
    scipy.misc.imsave('check.png', check)
    for i in range(SZ * SZ):
        check[i][i] = 0
    print(numpy.max(check), numpy.min(check))
    return 0

if __name__ == '__main__':
    sys.exit(main())
