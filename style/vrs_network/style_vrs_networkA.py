
import tensorflow as tf
import kaleido as kld
from vrs_network.style_vrs_network1 import *

##### VRS
class vrs( kld.tf.vrs , base ):

    ### BUILD
    def build( self , input , instance = None ):

        self.parsin , self.parsout = [] , []
        self.instance , self.iter = instance , 0

        if kld.chk.is_tsr( instance ):
            self.parsin = kld.tf.plchf( instance.shape , 'parsin' )

        input = ( input - 0.5 ) * 2.0
        input = self.reflect_padding( input , pad = 24 )

        with tf.variable_scope( 'Encoder' ):

            conv1 = self.conv_layer( input ,  32 , 3 , 1 , name = 'conv1' )
            conv2 = self.conv_layer( conv1 ,  64 , 3 , 2 , name = 'conv2' )
            conv3 = self.conv_layer( conv2 , 128 , 3 , 2 , name = 'conv3' )

        with tf.variable_scope( 'Residual' ):

            resid1 = self.residual_block(  conv3 , 3 , name = 'resid1' )
            resid2 = self.residual_block( resid1 , 3 , name = 'resid2' )
            resid3 = self.residual_block( resid2 , 3 , name = 'resid3' )

        with tf.variable_scope( 'Decoder' ):

            conv_t1 = self.conv_transpose_layer( resid3  , 64 , 3 , 2 , name = 'convt1' )
            conv_t2 = self.conv_transpose_layer( conv_t1 , 32 , 3 , 2 , name = 'convt2' )
            conv_t3 = self.conv_layer(           conv_t2 ,  3 , 3 , 1 , name = 'convt3' , relu = False )

        self.output = ( tf.nn.tanh( conv_t3 ) + 1.0 ) / 2.0

        if len( self.parsout ) > 0:
            self.parsout = tf.concat( self.parsout , axis = 0 )


