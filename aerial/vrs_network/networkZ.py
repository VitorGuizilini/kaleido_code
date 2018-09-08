
import tensorflow as tf
import kaleido as kld

def resconv( input , name , ch , ks , ops , post ):

    args1 = { 'post' : ops }
    args2 = { 'post' : post }

    input  = kld.tf.layer.conv2d( input  , [name,'A'] , ch , ks , 1 , **args1 )
    output = kld.tf.layer.conv2d( input  , [name,'B'] , ch , ks , 1 , **args1 )
    output = kld.tf.layer.conv2d( output , [name,'C'] , ch , ks , 1 , **args1 )
    output = kld.tf.layer.conv2d( output , [name,'D'] , ch , ks , 1 )

    return kld.tf.layer.conv2d( input + output , [name,'E'] , ch , ks , 2 , **args2 )

def resdeconv( input , name , ch , ks , shape , ops , post ):

    args1 = { 'post' : ops }
    args2 = { 'post' : post }

    input  = kld.tf.layer.trconv2d( input  , [name,'A'] , ch , ks , 1 , input , **args1 )
    output = kld.tf.layer.trconv2d( input  , [name,'B'] , ch , ks , 1 , input , **args1 )
    output = kld.tf.layer.trconv2d( output , [name,'C'] , ch , ks , 1 , input , **args1 )
    output = kld.tf.layer.trconv2d( output , [name,'D'] , ch , ks , 1 , input )

    return kld.tf.layer.trconv2d( input + output , [name,'E'] , ch , ks , 2 , shape , **args2 )

def network( input , label , lrate ):

    wgts_init = [ tf.truncated_normal_initializer , { 'mean' : 0.0 , 'stddev' : 0.01 } ]
    bias_init = tf.initializers.zeros
    kld.tf.layer.defaults( { 'wgts_init' : wgts_init , 'bias_init' : bias_init } )

    tfdrop = kld.tf.plchf( None , 'dropout' )
    tfphase = kld.tf.plchb( None , 'phase' )

    relu = ( tf.nn.relu )
    sigm = ( tf.nn.sigmoid )
    drop = ( tf.nn.dropout  , { 'keep_prob' : tfdrop } )
    pool = ( tf.nn.max_pool , { 'ksize'   : [ 1 , 2 , 2 , 1 ] ,
                                'strides' : [ 1 , 2 , 2 , 1 ] , 'padding' : 'SAME' } )
    norm = ( tf.contrib.layers.batch_norm ,
                              { 'epsilon' : 1e-5 , 'center' : True , 'scale' : True ,
                                'is_training' : tfphase } )

    ops_conv  , ops_deconv  = [ norm , relu ] , [ norm , relu ]
    post_conv , post_deconv = [ norm , relu ] , [ norm , relu ]

    res1conv = resconv( input    , 'res1conv' ,  64 , 5 , ops_conv , post_conv )
    res2conv = resconv( res1conv , 'res2conv' , 128 , 5 , ops_conv , post_conv )
    res3conv = resconv( res2conv , 'res3conv' , 256 , 3 , ops_conv , post_conv )
    latent   = resconv( res3conv , 'latent'   , 512 , 3 , ops_conv , post_conv )

    res3deconv = resdeconv( latent     , 'res3deconv' , 256 , 3 , res3conv , ops_deconv , post_deconv )
    res2deconv = resdeconv( res3deconv , 'res2deconv' , 128 , 3 , res2conv , ops_deconv , post_deconv )
    res1deconv = resdeconv( res2deconv , 'res1deconv' ,  64 , 3 , res1conv , ops_deconv , post_deconv )
    logits     = resdeconv( res1deconv , 'output'     ,   1 , 3 , input    , ops_deconv , [] )
    logits = tf.squeeze( logits , axis = 3 )
    output = tf.nn.sigmoid( logits )

    loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( labels = label , logits = logits ) )

    with tf.control_dependencies( kld.tf.update_ops() ):
        optim = tf.train.AdamOptimizer( lrate ).minimize( loss )

    return output , [ tfdrop , 0.3 , tfphase ] , loss , optim
