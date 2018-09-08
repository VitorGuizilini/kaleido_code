
import tensorflow as tf
import kaleido as kld
from networks.fst_network2 import *

##### NETWORK
class Network( NetworkBase ):

    ### __INIT__
    def __init__( self , args ):
        self.type , self.parsin , self.parsout , self.iter = None , [] , [] , 0
        self.args = args

    ### BUILD
    def build( self , image , type = None ):

        if type is not None:
            self.parsin = kld.tf.plchf( [ None , type ] , 'parsin' )

        self.type = type
        image_p = self.reflect_padding( image , pad = 24 )

        conv1 = self.conv_layer( image_p ,  32 , 3 , 1 , name = 'conv1' )
        conv2 = self.conv_layer(   conv1 ,  64 , 3 , 2 , name = 'conv2' )
        conv3 = self.conv_layer(   conv2 , 128 , 3 , 2 , name = 'conv3' )

        resid1 = self.residual_block(  conv3 , 3 , name = 'resid1' )
        resid2 = self.residual_block( resid1 , 3 , name = 'resid2' )
        resid3 = self.residual_block( resid2 , 3 , name = 'resid3' )

        conv_t1 = self.conv_tranpose_layer( resid3  , 64 , 3 , 2 , name = 'convt1' )
        conv_t2 = self.conv_tranpose_layer( conv_t1 , 32 , 3 , 2 , name = 'convt2' )

        conv_t3 = self.conv_layer( conv_t2 , 3 , 3 , 1 , relu = False , name = 'convt3' )

        preds = ( tf.nn.tanh( conv_t3 ) + 1 ) * ( 255.0 / 2.0 )
        preds = tf.squeeze( preds )
        preds = tf.clip_by_value( preds , 0.0 , 255.0 )

        if len( self.parsout ) > 0: self.parsout = tf.concat( self.parsout , axis = 1 )


        return preds , self.parsin , self.parsout
