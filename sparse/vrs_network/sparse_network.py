
import kaleido as kld
import tensorflow as tf

#############################################################################

### SMOOTH
def smooth( pred , label ):

    predX = kld.image.gradX( pred )
    predY = kld.image.gradY( pred )

    labelX = kld.image.gradX( label )
    labelY = kld.image.gradY( label )

    wgtsX = tf.exp( - tf.abs( labelX ) )
    wgtsY = tf.exp( - tf.abs( labelY ) )

    smoothX = tf.reduce_mean( tf.multiply( wgtsX , predX ) )
    smoothY = tf.reduce_mean( tf.multiply( wgtsY , predY ) )

    return smoothX + smoothY
