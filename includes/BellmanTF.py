import tensorflow as tf
import numpy as np


class BellmanTF():

    def oneShotMRPEvaluation(self, V, P, R, gamma):
        """Don't call it more than once over the same set of variables. Evaluate the operator whenever needed.
        It is only good for small MRP's with discount < 1, finite reward set, finite state set, 
        static transition probabilities. Caculates all the values of states.
        
        Arguments:
            V {np.variable} -- matrix of shape [n,1] where n is the number of states
            P {np.variable} -- matrix of [n,n] where each row is s and each column is s'. static transition probabilities
            R {np.variable} -- same as the shape of V. finite reward set
            gamma{np.constant}
        
        Returns:
            
        """

        dnom = tf.subtract(1, tf.multiply(P, gamma) )
        invDnom = tf.matrix_inverse(dnom)
        return V.assign(tf.matmul(invDnom, R))



