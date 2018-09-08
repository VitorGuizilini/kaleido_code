
import kaleido as kld
import tensorflow as tf
from source.networks.source_Network import *

pool = [ tf.nn.max_pool , { 'ksize'   : [ 1 , 3 , 3 , 1 ] ,
                            'strides' : [ 1 , 2 , 2 , 1 ] , 'padding' : 'SAME' } ]

### CONV
def conv( input , name , channels , ksizes , strides , ops1 = None , ops2 = None ):
    convs = []
    for ksize in ksizes:
        conv = kld.tf.layer.conv2d( input , name + str( ksize ) , channels , ksize , strides )
        convs.append( conv )
    if strides == 2:
        conv = kld.tf.layer.conv2d( input , name + 'MP' , channels , 1 , 1 )
        convs.append( kld.tf.apply_op( conv , pool ) )
    convs = tf.concat( convs , axis = 3 )
    convs = kld.tf.apply_op( convs , ops1 )
    convs = kld.tf.layer.conv2d( convs , name + 'OUT' , channels , 1 , 1 )
    convs = kld.tf.apply_op( convs , ops2 )
    return convs

### DECONV
def deconv( input , name , channels , ksizes , convs , ops1 = None , ops2 = None ):
    scale , deconvs = kld.tf.shape( convs ) , []
    for ksize in ksizes:
        deconv = kld.tf.layer.rsconv2d( input , name + str( ksize ) , channels , ksize , scale )
        deconvs.append( deconv )
    deconvs = tf.concat( deconvs , axis = 3 )
    deconvs = kld.tf.apply_op( deconvs , ops1 )
    deconvs = tf.concat( [ deconvs , convs ] , axis = 3 )
    deconvs = kld.tf.layer.rsconv2d( deconvs , name + 'OUT' , channels , 1 , scale )
    deconvs = kld.tf.apply_op( deconvs , ops2 )
    return deconvs

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
        both = [ norm , actv ]

        with tf.variable_scope( 'Placeholders' ):

            shapein , shapelbl = self.calc_shapes()
            self.input = kld.tf.plch( [ None ] + shapein  , 'Input' )
            self.label = kld.tf.plch( [ None ] + shapelbl , 'Label' )

        with tf.variable_scope( 'Network' ):

            input , label = self.input , self.label
            input = tf.expand_dims( input , axis = 3 )

            conv1   =   conv(   input ,   'conv1' ,  32 , [9,7,5] ,   2   , both , both )
            conv2   =   conv(   conv1 ,   'conv2' ,  32 , [7,5,3] ,   2   , both , both )
            conv3   =   conv(   conv2 ,   'conv3' ,  64 ,   [5,3] ,   2   , both , both )
            conv4   =   conv(   conv3 ,   'conv4' , 128 ,     [3] ,   2   , both , both )
            latent  =   conv(   conv4 ,  'latent' , 256 ,     [3] ,   2   , both , both )
            deconv4 = deconv(  latent , 'deconv4' , 128 ,     [3] , conv4 , both , both )
            deconv3 = deconv( deconv4 , 'deconv3' ,  64 ,     [3] , conv3 , both , both )
            deconv2 = deconv( deconv3 , 'deconv2' ,  32 ,     [3] , conv2 , both , both )
            deconv1 = deconv( deconv2 , 'deconv1' ,  32 ,     [3] , conv1 , both , both )
            output  = deconv( deconv1 ,  'output' ,   1 ,     [3] , input , both , sigm )

        with tf.variable_scope( 'Optimizer' ):

            self.output = output = tf.squeeze( output , axis = 3 )

            self.loss_rmse  = tf.reduce_mean( tf.square( output - label ) )
            self.loss = self.loss_rmse

            self.lrate = tf.placeholder( tf.float32 , None , 'LRate' )
            self.optim = tf.train.AdamOptimizer( learning_rate = self.lrate ,
                                    beta1 = 0.9 , beta2 = 0.999 , epsilon = 1e-8 )
            self.optim = self.optim.minimize( self.loss )

        return tf.Session()
