
import tensorflow as tf
import kaleido as kld

##### VRS
class vrs( kld.tf.vrs ):

    ### BUILD
    def build( self , loss , epoch , num_epochs , num_batches ):

        self.LRate = kld.mng.LRate( 'linear' , start = 1e-1 , finish = 1e-2 ,
                                     num_steps = num_epochs * num_batches ,
                                     step = epoch * num_batches )

        self.lrate = kld.tf.plchf( None , 'lrate' )
        optimizer = tf.train.MomentumOptimizer( self.lrate , momentum = 0.9 )
        self.run = optimizer.minimize( loss )
