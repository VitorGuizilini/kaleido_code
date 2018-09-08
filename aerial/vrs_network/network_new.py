
import tensorflow as tf
import kaleido as kld

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

def network( input , label , lrate ):

    wgts_init = [ tf.contrib.layers.xavier_initializer , { 'uniform' : False } ]
    bias_init = tf.initializers.zeros
    wgts_regl = [ tf.contrib.layers.l2_regularizer , { 'scale' : 1.0 } ]
    bias_regl = [ tf.contrib.layers.l2_regularizer , { 'scale' : 1.0 } ]

    kld.tf.layer.defaults( { 'wgts_init' : wgts_init , 'bias_init' : bias_init ,
                             'wgts_regl' : wgts_regl , 'bias_regl' : bias_regl } )

    tfdrop = kld.tf.plchf( None , 'dropout' )

    actv = tf.nn.relu
    sigm = tf.nn.sigmoid
    norm = [ tf.contrib.layers.instance_norm ,
                { 'epsilon' : 1e-5 , 'center' : True , 'scale' : True } ]
    drop = [ tf.nn.dropout ,
                { 'keep_prob' : tfdrop } ]

    both = [ norm , actv ]
    trip = [ norm , drop , actv ]

    conv1   =   conv(   input ,   'conv1' ,  32 , [5,3] ,   2   , both , both )
    conv2   =   conv(   conv1 ,   'conv2' ,  32 , [5,3] ,   2   , both , both )
    conv3   =   conv(   conv2 ,   'conv3' ,  64 ,   [3] ,   2   , both , both )
    conv4   =   conv(   conv3 ,   'conv4' , 128 ,   [3] ,   2   , both , both )
    latent  =   conv(   conv4 ,  'latent' , 256 ,   [3] ,   2   , both , both )
    deconv4 = deconv(  latent , 'deconv4' , 128 ,   [3] , conv4 , both , both )
    deconv3 = deconv( deconv4 , 'deconv3' ,  64 ,   [3] , conv3 , both , both )
    deconv2 = deconv( deconv3 , 'deconv2' ,  32 ,   [3] , conv2 , both , both )
    deconv1 = deconv( deconv2 , 'deconv1' ,  32 , [5,3] , conv1 , both , both )
    logits  = deconv( deconv1 ,  'output' ,   1 , [5,3] , input , both ,  []  )
    logits  = tf.squeeze( logits , axis = 3 )
    output  = tf.nn.sigmoid( logits )

    loss1 = - tf.reduce_mean( label * tf.log( output + 1e-6 ) + ( 1 - label ) * tf.log( 1 - output + 1e-6 ) )
    loss2 = tf.reduce_mean( kld.tf.regularization_losses() )
    loss = loss1 + 1e-3 * loss2

    optim = tf.train.AdamOptimizer( lrate ).minimize( loss )

    return output , [ tfdrop , 0.3 ] , loss , optim
