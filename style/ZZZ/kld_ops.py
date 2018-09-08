
import tensorflow as tf
import kaleido as kld

########################
### TOTAL VARIATION LOSS
def total_variation_loss( img ):

    b , h , w , d = kld.get_shape( img )

    x_tv_size = kld.toFloat( h * ( w - 1 ) * d )
    y_tv_size = kld.toFloat( ( h - 1 ) * w * d )
    b = kld.toFloat( b )

    x_tv = tf.nn.l2_loss( img[ : ,  : , 1: , : ] - img[ : , : , :w - 1 , : ] )
    y_tv = tf.nn.l2_loss( img[ : , 1: ,  : , : ] - img[ : , :h - 1 , : , : ] )

    loss = 2.0 * ( x_tv / x_tv_size
                 + y_tv / y_tv_size ) / b

    return loss

###############
### GRAM MATRIX
def gram_matrix( tensor ):

    b , h , w , c = kld.get_shape( tensor )
    chw = kld.toFloat( c * h * w )

    feats = tf.reshape( tensor , ( b , h * w , c ) )
    feats_T = tf.transpose( feats , perm = [ 0 , 2 , 1 ] )
    gram = tf.matmul( feats_T , feats ) / chw

    return gram

#############
### LOSS MSE
def loss_mse( tensor1 , tensor2 ):
    return tf.reduce_mean( tf.square( tensor1 - tensor2 ) )

#############
### LOSS MAE
def loss_mae( tensor1 , tensor2 ):
    return tf.reduce_mean( tf.abs( tensor1 - tensor2 ) )

#################
### LOSS KLDIV
def loss_kldiv( mu ):
    # KL_divergence = 0.5 * tf.reduce_sum( tf.square( mu ) + tf.square( sigma ) - tf.log( 1e-8 + tf.square( sigma ) ) - 1 , axis = -1 )
    # loss = tf.reduce_mean(KL_divergence)
    mu2 = tf.square( mu )
    loss = tf.reduce_mean( mu2 )
    return loss

################
### AE LOSS MSE
def ae_loss_mse( tensor , func ):
    return loss_mse( tensor , func( tensor ) )

################
### AE LOSS MAE
def ae_loss_mae( tensor , func ):
    return loss_mae( tensor , func( tensor ) )



