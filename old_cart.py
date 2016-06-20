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

def idx_iter(i_from, i_to, size):
    i = i_from
    while i % size != i_to % size:
        yield i % size
        i += 1

def field(row_from, row_to, col_from, col_to, size):
    row_from %= size
    row_to %= size
    col_from %= size
    col_to %= size
    field = numpy.zeros(((row_to - row_from) % size,
                         (col_to - col_from) % size, 2))
    for j, y in enumerate(idx_iter(col_from, col_to, size)):
        for i, x in enumerate(idx_iter(row_from, row_to, size)):
            field[i][j] = ((float(x), float(y)))
    return field

def cartesian_dist(src, tgt):
    src_x, src_y = src
    tgt_x, tgt_y = tgt
    dist = math.sqrt((src_x - tgt_x) ** 2 + (src_y - tgt_y) ** 2)
    return math.exp(-dist)

def toroidal_dist(lenfield):
    def toroidal_inner(src, tgt):
        src_x, src_y = src
        tgt_x, tgt_y = tgt
        dist = min((src_x - tgt_x) % lenfield, (tgt_x - src_x) % lenfield) ** 2.
        dist += min((src_y - tgt_y) % lenfield, (tgt_y - src_y) % lenfield) ** 2.
        # dist = ((src_x - tgt_x) % (lenfield / 1)) ** 2
        # dist += ((src_y - tgt_y) % (lenfield / 1)) ** 2
        return math.exp(-math.sqrt(dist))
    return toroidal_inner

def mat2cov(field, dist_metric):
    covsz = field.shape[0] ** 2
    cov = numpy.zeros((covsz, covsz))
    for i in range(covsz):
        src = field[i / field.shape[0]][i % field.shape[0]]
        for j in range(covsz):
            tgt = field[j / field.shape[0]][j % field.shape[0]]
            cov[i][j] = dist_metric(src, tgt)
    return cov

def cov2row(cov, x, y):
    SZ = int(math.sqrt(cov.shape[0]))
    row = x + SZ * y
    return cov[row].reshape((SZ, SZ))

def get_margins(size, margin, s_i):
    margin_sinister = margin
    margin_dexter = margin
    idx_from = s_i * size
    idx_to = (s_i + 1) * size
    idx_from -= margin_sinister
    idx_to += margin_dexter
    return (idx_from, idx_to, margin_sinister, margin_dexter)

def write_to_invcov(lenfield, size, margin, n_superblocks, out_invcov, offsets):
    i_offset, j_offset = offsets
    for s_i in range(n_superblocks):
        rlmargins = get_margins(size, margin, s_i)
        row_from, row_to, margin_left, margin_right = rlmargins
        if i_offset:
            row_from += size / 2
            row_to += size / 2
        for s_j in range(n_superblocks):
            tbmargins = get_margins(size, margin, s_j)
            col_from, col_to, margin_top, margin_bot = tbmargins
            if j_offset:
                col_from += size / 2
                col_to += size / 2
            sub_mat = field(row_from, row_to, col_from, col_to, lenfield)
            cov = mat2cov(sub_mat, toroidal_dist(lenfield))
            invcov = linalg.inv(cov)
            for r_i in range(size):
                for r_j in range(size):
                    invcov_row = (margin_top + r_i) * (size + margin * 2)
                    invcov_row += margin_left + r_j
                    r_orig = (((size / 2) * i_offset + s_i * size + r_i) % lenfield,
                              ((size / 2) * j_offset + s_j * size + r_j) % lenfield)
                    out_row = r_orig[0] * lenfield + r_orig[1]
                    for c_i in range(size):
                        for c_j in range(size):
                            invcov_col = (margin_top + c_i) * (size + margin * 2)
                            invcov_col += margin_left + c_j
                            val = invcov[invcov_row][invcov_col]
                            c_orig = (((size / 2) * i_offset + s_i * size + c_i) % lenfield,
                                      ((size / 2) * j_offset + s_j * size + c_j) % lenfield)
                            out_col = c_orig[0] * lenfield + c_orig[1]
                            out_invcov[out_row][out_col] = val

def quick_inv(field, size, margin):
    out_invcov = numpy.zeros((len(field) ** 2,) * 2)
    # Tile the field with rectangles size x size.
    n_superblocks = (len(field) / size) + (1 if (len(field) % size) else 0)
    for i_offset in range(2):
        for j_offset in range(2):
            offsets = (i_offset, j_offset)
            write_to_invcov(len(field), size, margin, n_superblocks, out_invcov,
                            offsets)
    return out_invcov

def main():
    SZ = 32
    field = mat2d(SZ)
    cov = mat2cov(field, toroidal_dist(SZ))
    scipy.misc.imsave('cov.png', cov)
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
