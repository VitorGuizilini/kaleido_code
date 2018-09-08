
import tensorflow as tf
import kaleido as kld

### APPROX
def approx( input , name ):

    channels = [ 100 , 200 ]
    shape = kld.tf.shape( input )
    layer = tf.reshape( input , [ shape[0] , shape[-1] ] )

    for k , ch in enumerate( channels ):
        layer = kld.tf.layer.dense( layer , name + str(k) , ch )
        layer = tf.nn.relu( layer )
    layer = kld.tf.layer.dense( layer , name + 'out' , shape[-1] )

    return tf.reshape( layer , shape )

##### VRS
class vrs( kld.tf.vrs ):

    ### BUILD
    def build( self , input , name ):

        mu , sq = tf.nn.moments( input , [ 1 , 2 ] , keep_dims = True )

        self.mu = approx( mu , name + 'mu' )
        self.sq = approx( sq , name + 'sq' )
        self.sq = tf.square( self.sq )
