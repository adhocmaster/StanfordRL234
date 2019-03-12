import tensorflow as tf
import numpy as np


class RLUtilsTF():


    def createRowVec( self, n:int, initialVals=0., dtype=None, name=None ):
        """[summary]
        
        Arguments:
            n {int} -- [description]
        
        Keyword Arguments:
            initialVals {[type]} -- [description] (default: {0.})
            dtype {[type]} -- [description] (default: {None})
            name {[type]} -- [description] (default: {None})
        
        Returns:
            [type] -- Row vector with a initial value 0.
        """


        if dtype is None:
            return tf.Variable(tf.fill( [n, 1], initialVals ), name=name )
        else:
            return tf.Variable(
                        tf.cast( tf.fill( [n, 1], initialVals ), name=name , dtype=dtype )
            )
    

    def createRowVecfromNumpyVector( self, npV:np.ndarray, dtype=None, name=None ):

        if dtype is not None:
            npV = npV.astype( dtype )
        
        return tf.Variable( tf.reshape(npV, [-1,1]), name=name )

    def createRowVecfromNumpyMatrix( self, npV, dtype=None, name=None ):

        if dtype is not None:
            npV = npV.astype( dtype )
            
        return tf.Variable( npV, name=name  )
    

    def createRandomProbabiltyMatrix( self, n, dtype=tf.float32 ):

        with tf.name_scope( 'rlUtilsP' ):

            vec = tf.get_variable("temp_vec", shape=[n], initializer=tf.zeros_initializer(), dtype=dtype )
            
            ta = tf.TensorArray( dtype=dtype, size=n )

            loopVars = ( 0, ta )

            condition = lambda i, _: i < n

            def body( i, ta ): 
                # added operators inside loop as tensorflow does not evaluate operators 
                # declared outside the loop more than once
                randomizeVec = vec.assign( tf.random_uniform( shape=[n], seed=39, dtype=dtype ) )
                softmaxVec = tf.divide( randomizeVec, tf.reduce_sum( randomizeVec ) )

                # control dependencies cannot ensure re-evaluation.
                with tf.control_dependencies([randomizeVec, softmaxVec]):
                    return ( i + 1, ta.write(i, softmaxVec ) ) 
            

            _, ta_final = tf.while_loop( condition, body, loopVars )

            return ta_final.stack()




