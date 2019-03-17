import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from core.deep_q_learning import DQN
from q1_schedule import LinearExploration, LinearSchedule

from configs.q2_linear import config

import pprint


class Linear(DQN):
    """
    Implement Fully Connected with Tensorflow
    """
    def add_placeholders_op(self):
        """
        Adds placeholders to the graph

        These placeholders are used as inputs to the rest of the model and will be fed
        data during training.
        """
        # this information might be useful
        state_shape = list(self.env.observation_space.shape)

        ##############################################################
        """
        TODO: 
            Add placeholders:
            Remember that we stack 4 consecutive frames together.
                - self.s: batch of states, type = uint8
                    shape = (batch_size, img height, img width, nchannels x config.state_history)
                - self.a: batch of actions, type = int32
                    shape = (batch_size)
                - self.r: batch of rewards, type = float32
                    shape = (batch_size)
                - self.sp: batch of next states, type = uint8
                    shape = (batch_size, img height, img width, nchannels x config.state_history)
                - self.done_mask: batch of done, type = bool
                    shape = (batch_size)
                - self.lr: learning rate, type = float32
        
        (Don't change the variable names!)
        
        HINT: 
            Variables from config are accessible with self.config.variable_name.
            Check the use of None in the dimension for tensorflow placeholders.
            You can also use the state_shape computed above.
        """
        ##############################################################
        ################YOUR CODE HERE (6-15 lines) ##################
        obShape = env.observation_space.shape
        stateShape = [config.batch_size, obShape[0], obShape[1], obShape[2] * config.state_history ]
        
        self.s = tf.placeholder(tf.uint8, obShape)
        self.a = tf.placeholder(tf.int32, shape=[config.batch_size])
        self.r = tf.placeholder(tf.float32, shape=[config.batch_size])
        self.sp = tf.placeholder(tf.uint8, obShape)
        self.done_mask = tf.placeholder(tf.bool, shape=[config.batch_size])
        self.gamma = config.gamma
        self.alpha = 0.5

        pass

        ##############################################################
        ######################## END YOUR CODE #######################


    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels x config.state_history)
                *** This is not a fucking batch. It's a single state of shape (img height, img width, nchannels). Both the batch_size and state history missing.
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful

        print( "get_q_values_op state")
        print( state )
        num_actions = self.env.action_space.n

        ##############################################################
        """
        TODO: 
            Implement a fully connected with no hidden layer (linear
            approximation with bias) using tensorflow.

        HINT: 
            - You may find the following functions useful:
                - tf.layers.flatten
                - tf.layers.dense

            - Make sure to also specify the scope and reuse
        """
        ##############################################################
        ################ YOUR CODE HERE - 2-3 lines ################## 

        with tf.variable_scope(scope, reuse = reuse):
            
            # weight and bias
            #w = tf.get_variable( name = 'weight', initializer=tf.zeros_initializer())
            #b = tf.get_variable( name = 'bias', initializer=tf.zeros_initializer())
            X = self.getStateActionsOfAState( state )

            print( "get_q_values_op X")
            print( X )

            # stateActionOut will have [batchSize * num_actions, 1] shape 
            stateActionOut = tf.layers.dense(
                                    name='fcc',
                                    inputs=X, 
                                    units=tf.float32,
                                    activation=None,
                                    use_bias=True
                                )

        
        pass

        ##############################################################
        ######################## END YOUR CODE #######################

        print( "get_q_values_op stateActionOut")
        print( stateActionOut )
        # stateActionOut will have [batchSize * num_actions, 1] shape. we convert it to  (batch_size, num_actions)
        return tf.reshape( stateActionOut, [1, -1] )


    def getStateActionsOfAState( self, state ):
        
        num_actions = self.env.action_space.n
        oneHotActions = self.getOneHotOfActions()
        extendedState = state[None, :]
        flattenedXWithOneBatch = layers.flatten( extendedState )

        # [ num_actions, flatsize]
        repeatedStates = tf.tile( flattenedXWithOneBatch, [num_actions, 1] )

        print( repeatedStates )
        print( oneHotActions )
        X = tf.concat( [repeatedStates, oneHotActions], axis = 1 )

        return X


    def getStateActionsOfABatch( self, states ):
        num_actions = self.env.action_space.n
        oneHotActions = self.getOneHotOfActions()
        flattenedStates = layers.flatten( states ) # [batchSize, stateSize]
        batchSize = tf.shape(flattenedStates)[0]
        stateSize = tf.shape(flattenedStates)[1]
        # now each state will be put into a new dimension so that we can tile that dimension num_actions times
        # [batchSize, 1, stateSize]
        expandedStates = tf.expand_dims(flattenedStates, 1)
        # now we will get for each state, num_actions copies
        # [batchSize, num_actions, stateSize]
        tiledStates = tf.tile( expandedStates, [1, num_actions, 1] )
        # [batchSize * num_actions, stateSize]
        repeatedStates = tf.reshape( tiledStates, [-1, stateSize] )

        # [batchSize * num_actions] action set is repeated for each state
        batchActions = tf.tile( oneHotActions, [batchSize, 1])

        print( states )
        print( flattenedStates )
        print( expandedStates )
        print( tiledStates )
        print( repeatedStates )
        # [batchSize, (stateSize + onehotsize) * num_actions]
        X = tf.concat( [repeatedStates, batchActions], axis = 1 )

        return X
        


    def getOneHotOfActions( self ):
        
        num_actions = self.env.action_space.n
        return tf.one_hot( indices = list( range(num_actions) ), depth = num_actions )

    def add_update_target_op(self, q_scope, target_q_scope):
        """
        update_target_op will be called periodically 
        to copy Q network weights to target Q network

        Remember that in DQN, we maintain two identical Q networks with
        2 different sets of weights. In tensorflow, we distinguish them
        with two different scopes. If you're not familiar with the scope mechanism
        in tensorflow, read the docs
        https://www.tensorflow.org/programmers_guide/variable_scope

        Periodically, we need to update all the weights of the Q network 
        and assign them with the values from the regular network. 
        Args:
            q_scope: (string) name of the scope of variables for q
            target_q_scope: (string) name of the scope of variables
                        for the target network
        """
        ##############################################################
        """
        TODO: 
            Add an operator self.update_target_op that for each variable in
            tf.GraphKeys.GLOBAL_VARIABLES that is in q_scope, assigns its
            value to the corresponding variable in target_q_scope

        HINT: 
            You may find the following functions useful:
                - tf.get_collection
                - tf.assign
                - tf.group (the * operator can be used to unpack a list)

        (be sure that you set self.update_target_op)
        """
        ##############################################################
        ################### YOUR CODE HERE - 5-10 lines #############

        # sourceCollection = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES, scope = q_scope )
         # destCollection = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES, scope=target_q_scope )
        sourceCollection = tf.global_variables( scope = q_scope )
        destCollection = tf.global_variables( scope = target_q_scope )
        ops = []
        for i in range( len( sourceCollection ) ):
            ops.append( tf.assign(destCollection[i], sourceCollection[i] ) )
        '''
        w_update = tf.assign( destCollection.)
        self.update_target_op = 

        with tf.variable_scope( q_scope ):
            s_w = tf.get_variable
        '''

        self.update_target_op = tf.group( *ops )
        
        pass

        ##############################################################
        ######################## END YOUR CODE #######################


    def add_loss_op(self, q, target_q):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions) #next state batch?
        """
        # you may need this variable
        num_actions = self.env.action_space.n

        ##############################################################
        """
        TODO: 
            The loss for an example is defined as:
                Q_samp(s) = r if done
                          = r + gamma * max_a' Q_target(s', a')
                loss = (Q_samp(s) - Q(s, a))^2 
        HINT: 
            - Config variables are accessible through self.config
            - You can access placeholders like self.a (for actions)
                self.r (rewards) or self.done_mask for instance
            - You may find the following functions useful
                - tf.cast
                - tf.reduce_max
                - tf.reduce_sum
                - tf.one_hot
                - tf.squared_difference
                - tf.reduce_mean
        """
        ##############################################################
        ##################### YOUR CODE HERE - 4-5 lines #############

        # compute val
        targetMax = tf.add( self.r, tf.multiply( tf.reduce_max( target_q, axis = 1 ), self.gamma ) )
        
        # shape = [batchSize]
        targetV = tf.where( self.done_mask, self.r, targetMax )

        # make each state value repeatable. [batchSize, 1]
        expandStatesVs = tf.expand_dims( targetV, 1 )
        # [batchSize, numActions] V is repeated on axis 1, or per row.
        targetVA = tf.tile( expandStatesVs, [1, num_actions] )

        sqDifElements = tf.squared_difference( q, targetVA ) # TD(0) error squared

        self.loss = tf.reduce_mean( sqDifElements )

        pass

        ##############################################################
        ######################## END YOUR CODE #######################

    def GradCapper( self, gradient ):
        return tf.clip_by_norm( gradient, self.config.clip_val )

    def add_optimizer_op(self, scope):
        """
        Set self.train_op and self.grad_norm
        Args:
            scope: (string) scope name, that specifies if target network or not
        """

        ##############################################################
        """
        TODO: 
            1. get Adam Optimizer
            2. compute grads with respect to variables in scope for self.loss
            3. if self.config.grad_clip is True, then clip the grads
                by norm using self.config.clip_val 
            4. apply the gradients and store the train op in self.train_op
                (sess.run(train_op) must update the variables)
            5. compute the global norm of the gradients (which are not None) and store 
                this scalar in self.grad_norm

        HINT: you may find the following functions useful
            - tf.get_collection
            - optimizer.compute_gradients
            - tf.clip_by_norm
            - optimizer.apply_gradients
            - tf.global_norm
             
             you can access config variables by writing self.config.variable_name
        """
        ##############################################################
        #################### YOUR CODE HERE - 8-12 lines #############

        with tf.variable_scope( scope ):
            optimizer = tf.train.AdamOptimizer( learning_rate=0.001 )

            gvs = optimizer.compute_gradients(self.loss)

            if self.config.grad_clip:
                gvs = [ ( self.GradCapper( gv[0] ), gv[1] ) for gv in gvs ]
            
            gradients, variables = zip( *gvs ) # fucking unzip

            self.grad_norm = tf.global_norm( gradients )

            self.train_op = optimizer.apply_gradients( gvs )

        pass
        
        ##############################################################
        ######################## END YOUR CODE #######################
    


if __name__ == '__main__':
    env = EnvTest((5, 5, 1))

    pp = pprint.PrettyPrinter( width=4 )

    pp.pprint( config )
    

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = Linear(env, config)
    model.run(exp_schedule, lr_schedule)
