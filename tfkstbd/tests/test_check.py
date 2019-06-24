# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from collections import Counter
from ..check import mean_waste, estimate_buckets, estimate_batches


class TestMeaningWaste(unittest.TestCase):
    def testNormal(self):
        source = Counter({1: 5, 2: 2, 3: 1})
        result = mean_waste(source)
        self.assertEqual(0.5, result)


class TestEstimateBuckets(unittest.TestCase):
    def testNormal(self):
        source = Counter({
            255: 16, 256: 15, 257: 20, 258: 16, 259: 17, 260: 15, 261: 15, 262: 12, 263: 13, 264: 13, 265: 11, 266: 9,
            267: 8, 268: 9, 269: 7, 270: 9, 271: 7, 272: 6, 273: 5, 274: 6, 275: 5, 276: 4, 277: 4, 278: 4, 279: 4,
            280: 4, 281: 5, 282: 3, 283: 3, 284: 3, 285: 3, 286: 2, 287: 3, 288: 2, 289: 2, 290: 3, 291: 2, 292: 1,
            293: 2, 294: 1, 295: 2, 296: 1, 297: 1, 298: 1, 300: 1, 301: 1, 303: 1, 304: 1, 305: 1, 311: 1
        })
        result = estimate_buckets(source)
        self.assertListEqual([262, 268, 274, 281, 287, 294, 301], result)


class TestEstimateBatches(unittest.TestCase):
    def testNormal(self):
        source_lens = Counter({
            255: 16, 256: 15, 257: 20, 258: 16, 259: 17, 260: 15, 261: 15, 262: 12, 263: 13, 264: 13, 265: 11, 266: 9,
            267: 8, 268: 9, 269: 7, 270: 9, 271: 7, 272: 6, 273: 5, 274: 6, 275: 5, 276: 4, 277: 4, 278: 4, 279: 4,
            280: 4, 281: 5, 282: 3, 283: 3, 284: 3, 285: 3, 286: 2, 287: 3, 288: 2, 289: 2, 290: 3, 291: 2, 292: 1,
            293: 2, 294: 1, 295: 2, 296: 1, 297: 1, 298: 1, 300: 1, 301: 1, 303: 1, 304: 1, 305: 1, 311: 1
        })
        source_bucks = [262, 268, 274, 281, 287, 294, 301]
        result = estimate_batches(source_lens, source_bucks, 1024)
        self.assertListEqual([1.177, 1.15, 1.124, 1.096, 1.074, 1.048, 1.024, 1.0], result)
