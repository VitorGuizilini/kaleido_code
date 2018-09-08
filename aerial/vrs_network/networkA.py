
import tensorflow as tf
import kaleido as kld

def network( input , label , lrate ):

    wgts_init = [ tf.truncated_normal_initializer , { 'mean' : 0.0 , 'stddev' : 0.1 } ]
    bias_init = [ tf.truncated_normal_initializer , { 'mean' : 0.0 , 'stddev' : 0.1 } ]

    kld.tf.layer.defaults( { 'wgts_init' : wgts_init , 'bias_init' : bias_init } )

    tfdrop = kld.tf.plchf( None , 'dropout' )

    actv = tf.nn.relu
    sigm = tf.nn.sigmoid
    pool = [ tf.nn.max_pool ,
                { 'ksize'   : [ 1 , 2 , 2 , 1 ] ,
                  'strides' : [ 1 , 2 , 2 , 1 ] , 'padding' : 'SAME' } ]
    drop = [ tf.nn.dropout ,
                { 'keep_prob' : tfdrop } ]

    args_conv   = { 'post' : [ pool , actv ] }
    args_latent = { 'post' : [ actv , drop ] }
    args_deconv = { 'post' : [ ] }
    args_output = { 'post' : [ ] }

    conv1  = kld.tf.layer.conv2d( input , ''  ,  64 , 5 , 1 , wgts_name = 'Variable'   , bias_name = 'Variable_1' , **args_conv )
    conv2  = kld.tf.layer.conv2d( conv1 , ''  , 128 , 5 , 1 , wgts_name = 'Variable_2' , bias_name = 'Variable_3' , **args_conv )
    conv3  = kld.tf.layer.conv2d( conv2 , ''  , 256 , 5 , 1 , wgts_name = 'Variable_4' , bias_name = 'Variable_5' , **args_conv )
    latent = kld.tf.layer.conv2d( conv3 , ''  , 256 , 3 , 2 , wgts_name = 'Variable_6' , bias_name = 'Variable_7' , **args_latent )

    deconv3 = kld.tf.layer.trconv2d( latent  , '' , 256 , 5 , 2 , conv3 , wgts_name = 'Variable_8'  , bias_name = 'Variable_9'  , **args_deconv )
    deconv2 = kld.tf.layer.trconv2d( deconv3 , '' , 128 , 5 , 2 , conv2 , wgts_name = 'Variable_10' , bias_name = 'Variable_11' , **args_deconv )
    deconv1 = kld.tf.layer.trconv2d( deconv2 , '' ,  64 , 5 , 2 , conv1 , wgts_name = 'Variable_12' , bias_name = 'Variable_13' , **args_deconv )
    logits  = kld.tf.layer.trconv2d( deconv1 , ''  ,  1 , 5 , 2 , input , wgts_name = 'Variable_14' , bias_name = 'Variable_15' , **args_output )
    logits  = tf.squeeze( logits , axis = 3 )
    output  = tf.nn.sigmoid( logits )

#    loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( labels = label , logits = logits ) )
    loss = - tf.reduce_mean( label * tf.log( output + 1e-6 ) + ( 1 - label ) * tf.log( 1 - output + 1e-6 ) )
    optim = tf.train.AdamOptimizer( lrate ).minimize( loss )

    return output , [ tfdrop , 0.4 ] , loss , optim
