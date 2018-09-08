
import tensorflow as tf
import kaleido as kld
from networks.network1 import *

##### NETWORK
class Network:

    ### __INIT__
    def __init__( self , args ):
        self.args = args

    ### BUILD
    def build( self , image ):

        image = image / 255.0
        image_p = reflect_padding( image )

        conv1 = conv_layer( image_p ,  32 , 3 , 1 , name = 'conv1' )
        conv2 = conv_layer(   conv1 ,  64 , 3 , 2 , name = 'conv2' )
        conv3 = conv_layer(   conv2 , 128 , 3 , 2 , name = 'conv3' )

        resid1 = residual_block(  conv3 , 3 , name = 'resid1' )
        resid2 = residual_block( resid1 , 3 , name = 'resid2' )
        resid3 = residual_block( resid2 , 3 , name = 'resid3' )
        resid4 = residual_block( resid3 , 3 , name = 'resid4' )
        resid5 = residual_block( resid4 , 3 , name = 'resid5' )

        conv_t1 = conv_tranpose_layer( resid5  , 64 , 3 , 2 , name = 'convt1' )
        conv_t2 = conv_tranpose_layer( conv_t1 , 32 , 3 , 2 , name = 'convt2' )

        conv_t3 = conv_layer( conv_t2 , 3 , 3 , 1 , relu = False , name = 'convt3' )

        preds = ( tf.nn.tanh( conv_t3 ) + 1 ) * ( 255.0 / 2.0 )
        return preds
