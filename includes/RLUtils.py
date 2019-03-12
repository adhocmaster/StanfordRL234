import tensorflow as tf
import numpy as np


class RLUtilsTF():


    def createRowVec( self, n:int, initialVals=0., type=None, name=None ):
        """[summary]
        
        Arguments:
            n {int} -- [description]
        
        Keyword Arguments:
            initialVals {[type]} -- [description] (default: {0.})
            type {[type]} -- [description] (default: {None})
            name {[type]} -- [description] (default: {None})
        
        Returns:
            [type] -- Row vector with a initial value 0.
        """


        if type is None:
            return tf.Variable(tf.fill( [n, 1], initialVals ), name=name )
        else:
            return tf.Variable(
                        tf.cast( tf.fill( [n, 1], initialVals ), name=name , dtype=type )
            )
    

    def createRowVecfromNumpyVector( self, npV:np.ndarray, type=None, name=None ):

        if type is not None:
            npV = npV.astype( type )
        
        return tf.Variable( tf.reshape(npV, [-1,1]), name=name )

    def createRowVecfromNumpyMatrix( self, npV, type=None, name=None ):

        if type is not None:
            npV = npV.astype( type )
            
        return tf.Variable( npV, name=name  )
    
    def createRandomProbabiltyMatrix( self, n ):

        with tf.name_scope( 'rlUtilsP' ):

            vec = tf.get_variable("temp_vec", shape=[n], initializer=tf.zeros_initializer() )
            randomizeVec = vec.assign( tf.random_uniform( shape=[n], seed=39 ) )
            softmaxVec = tf.divide( randomizeVec, tf.reduce_sum( randomizeVec ) )

            ta = tf.TensorArray( dtype=tf.float32, size=n )

            loopVars = ( 0, ta )

            condition = lambda i, _: i < n

            body = lambda i, ta: ( i + 1, ta.write(i, softmaxVec ) )

            m, ta_final = tf.while_loop( condition, body, loopVars )

            return ta_final.stack()




