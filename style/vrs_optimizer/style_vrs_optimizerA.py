
import tensorflow as tf
import kaleido as kld

##### VRS
class vrs( kld.tf.vrs ):

    ### BUILD
    def build( self , loss , epoch , num_epochs , num_batches , vars = None ):

        vars = kld.tf.trainable_vars( vars )
        grads = tf.gradients( loss , vars )

        self.LRate = kld.mng.LRate( 'linear' , start = 1e-3 , finish = 1e-4 ,
                                     num_steps = num_epochs * num_batches ,
                                     step = epoch * num_batches )

        self.lrate = kld.tf.plchf( None , 'lrate' )
        optimizer = tf.train.AdamOptimizer( self.lrate )
        self.run = optimizer.apply_gradients( zip( grads , vars ) )



