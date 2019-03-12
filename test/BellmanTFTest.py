# Path hack.
import sys, os
sys.path.insert(0, os.path.abspath('..'))

import unittest
import includes.BellmanTF as BellmanTF
import includes.RLUtils as RLUtils

import tensorflow as tf
import numpy as np

class BellmanTFTest( unittest.TestCase ):

    def setUp( self ):

        self.bellmanTF = BellmanTF()
        self.rlUtils = RLUtils()
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        self.sessConfig=tf.ConfigProto(gpu_options=self.gpu_options)
        self.dtype = tf.float64
        pass


    def testOneShotMRPEvaluation( self ):

        with tf.variable_scope( 'testOneShotMRPEvaluation' ):
            V = self.rlUtils.createRowVec(5, name='V', dtype=self.dtype)
            R = tf.random_normal([5,1], mean = 0, stddev=1.0, seed=39, name='reward', dtype=self.dtype )
            P = self.rlUtils.createRandomProbabiltyMatrix( 5, dtype=self.dtype )
            gamma = 0.5
            bellman = BellmanTF()
            updateV = bellman.oneShotMRPEvaluation( V, P, R, gamma )
        
        with tf.Session( config=self.sessConfig ) as sess:
            sess.run(tf.global_variables_initializer())
            V_out = sess.run( [updateV] )
            self.assertNotEqual( V_out, np.zeros( shape=(5,1), dtype ))

if __name__ == '__main__':
    unittest.main()