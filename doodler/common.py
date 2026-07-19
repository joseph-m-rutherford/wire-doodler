#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

import numpy as np

Index = np.uint64
Integer = np.int64
Real = np.float64

def real_equality(a: Real, b: Real, tolerance: Real) -> bool:
    '''For values near the origin, use absolute comparison; otherwise do relative comparison'''
    if abs(a) < tolerance and abs(b) < tolerance:
        return True
    else:
        return abs(a-b)/max(abs(a),abs(b)) < tolerance
