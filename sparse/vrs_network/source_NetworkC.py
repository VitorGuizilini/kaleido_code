
import kaleido as kld
import tensorflow as tf
from source.networks.source_Network import *

### EXPAND
def expand( input ):

    shape = kld.tf.shape( input )
    coordH = tf.range( 0.0 , shape[1] , 1.0 )
    coordW = tf.range( 0.0 , shape[2] , 1.0 )

    coordH = tf.expand_dims( coordH , axis = 1 )
    coordH = tf.tile( coordH , ( 1 , shape[2] ) )
    coordH = tf.expand_dims( coordH , axis = 0 )
    coordH = tf.tile( coordH , ( shape[0] , 1 , 1 ) )
    coordH = tf.expand_dims( coordH , axis = 3 )

    coordW = tf.expand_dims( coordW , axis = 0 )
    coordW = tf.tile( coordW , ( shape[1] , 1 ) )
    coordW = tf.expand_dims( coordW , axis = 0 )
    coordW = tf.tile( coordW , ( shape[0] , 1 , 1 ) )
    coordW = tf.expand_dims( coordW , axis = 3 )

    return tf.concat( [ input , coordH , coordW ] , axis = 3 )

### CONV
def conv( input , name , channels , ksize , strides , **args ):
    input = expand( input )
    out = kld.tf.layer.conv2d( input , name , channels , ksize , strides , **args )
    return out

### DECONV
def deconv( input , name , channels , ksize , add , concat = True , **args ):
    input = expand( input )
    out = kld.tf.layer.rsconv2d(   input , name , channels , ksize , add , **args )
    if concat: out = tf.concat( [ out , add ] , axis = 3 )
    return out

##### NETWORK
class Network:

    ### PREPARE
    def prepare( self ):

        wgts_init = [ tf.contrib.layers.xavier_initializer , { 'uniform' : False } ]
        bias_init = tf.initializers.zeros

        kld.tf.layer.defaults( { 'wgts_init' : wgts_init , 'bias_init' : bias_init } )

        actv = tf.nn.elu
        sigm = tf.nn.sigmoid
        norm = [ tf.contrib.layers.instance_norm , { 'epsilon' : 1e-5 , 'center' : True , 'scale' : True } ]

        args_conv   = { 'post' : [ norm , actv ] }
        args_deconv = { 'post' : [ norm , actv ] }
        args_output = { 'post' : [ sigm ] }

        with tf.variable_scope( 'Placeholders' ):

            shapein , shapelbl = self.calc_shapes()
            self.input = kld.tf.plch( [ None ] + shapein  , 'Input' )
            self.label = kld.tf.plch( [ None ] + shapelbl , 'Label' )

        with tf.variable_scope( 'Network' ):

            input , label = self.input , self.label
            input = tf.expand_dims( input , axis = 3 )

            conv1   =   conv(   input ,   'conv1' ,  32 , 3 ,   2   , **args_conv   )
            conv2   =   conv(   conv1 ,   'conv2' ,  64 , 3 ,   2   , **args_conv   )
            conv3   =   conv(   conv2 ,   'conv3' , 128 , 3 ,   2   , **args_conv   )
            conv4   =   conv(   conv3 ,   'conv4' , 256 , 3 ,   2   , **args_conv   )
            latent  =   conv(   conv4 ,  'latent' , 512 , 3 ,   2   , **args_conv   )
            deconv4 = deconv(  latent , 'deconv4' , 256 , 3 , conv4 , **args_deconv )
            deconv3 = deconv( deconv4 , 'deconv3' , 128 , 3 , conv3 , **args_deconv )
            deconv2 = deconv( deconv3 , 'deconv2' ,  64 , 3 , conv2 , **args_deconv )
            deconv1 = deconv( deconv2 , 'deconv1' ,  32 , 3 , conv1 , **args_deconv )
            output  = deconv( deconv1 ,  'output' ,   1 , 3 , input , **args_output , concat = False )

            print( input.shape )
            print( conv1.shape )
            print( conv2.shape )
            print( conv3.shape )
            print( conv4.shape )

        with tf.variable_scope( 'Optimizer' ):

            self.output = output = tf.squeeze( output , axis = 3 )

            self.loss_rmse  = tf.reduce_mean( tf.square( output - label ) )
            self.loss = self.loss_rmse

            self.lrate = tf.placeholder( tf.float32 , None , 'LRate' )
            self.optim = tf.train.AdamOptimizer( learning_rate = self.lrate ,
                                    beta1 = 0.9 , beta2 = 0.999 , epsilon = 1e-8 )
            self.optim = self.optim.minimize( self.loss )

        return tf.Session()
