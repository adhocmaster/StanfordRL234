# Path hack.
import sys, os
sys.path.insert(0, os.path.abspath('..'))

import unittest
import includes.BellmanTF as BellmanTF
import includes.RLUtils as RLUtils

import tensorflow as tf

class BellmanTFTest( unittest.TestCase ):

    def setUp( self ):

        self.bellmanTF = BellmanTF()
        self.rlUtils = RLUtils()
        pass


    def testOneShotMRPEvaluation( self ):

        V = self.rlUtils.createRowVec(5, name='V')
        R = tf.random_normal([5,1], mean = 0, stddev=1.0, seed=39, name='reward')

if __name__ == '__main__':
    unittest.main()