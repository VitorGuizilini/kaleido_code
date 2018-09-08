
import tensorflow as tf
import kaleido as kld

##### VRS
class vrs( kld.tf.vrs ):

    ### BUILD
    def build( self , net ):

        self.full = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(
                        labels = net.label , logits = net.logits ) )

