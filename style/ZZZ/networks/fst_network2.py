
import tensorflow as tf
import kaleido as kld

### APPROX
def approx( input , name , iter ):
    out , channels = input.shape[1] , [ 200 , 400 ]
    for k , ch in enumerate( channels ):
        input = kld.tf.layer.dense( input , name + str(iter) + str(k) , ch )
        input = tf.nn.relu( input )
    return kld.tf.layer.dense( input , name + str(iter) + 'out' , out )

class NetworkBase:

    ### CONV LAYER
    def conv_layer( self , net , num_filters , filter_size , strides ,
                    padding = 'SAME' , relu = True , name = None ):

        weights_init = self.conv_init_vars( net , num_filters , filter_size , name = name )
        strides_shape = [ 1 , strides , strides , 1 ]

        net = tf.nn.conv2d( net , weights_init , strides_shape , padding = padding )
        net = self.instance_norm( net , name = name )
        if relu: net = tf.nn.relu( net )

        return net

    ### CONV TRANSPOSE LAYER
    def conv_tranpose_layer( self , net , num_filters , filter_size , strides ,
                             padding = 'SAME' , relu = True , name = None ):

        weights_init = self.conv_init_vars( net , num_filters , filter_size , name = name , transpose = True )
        strides_shape = [ 1 , strides , strides , 1 ]

        batch , rows , cols , channels = kld.tf.shape( net )
        new_rows, new_cols = rows * strides , cols * strides
        new_shape = tf.stack( [ batch , new_rows , new_cols , num_filters ] )

        net = tf.nn.conv2d_transpose( net , weights_init , new_shape , strides_shape , padding = padding )
        net = self.instance_norm( net , name = name )
        if relu: net = tf.nn.relu( net )

        return net

    ### RESIDUAL BLOCK
    def residual_block( self , net , filter_size = 3 , name = None ):

        batch , rows , cols , channels = kld.tf.shape( net )
        tmp =  self.conv_layer( net , 128 , filter_size , 1 , padding = 'VALID' , relu = True  , name = name + '_1' )
        return self.conv_layer( tmp , 128 , filter_size , 1 , padding = 'VALID' , relu = False , name = name + '_2' ) \
                        + tf.slice( net , [ 0 , 2 , 2 , 0 ] , [ batch , rows - 4 , cols - 4 , channels ] )

    ### CONV INIT VARS
    def conv_init_vars( self , net , out_channels , filter_size ,
                        transpose = False , name = None ):

        in_channels = net.get_shape().as_list()[3]

        if not transpose:
            weights_shape = [ filter_size , filter_size , in_channels , out_channels ]
        else:
            weights_shape = [ filter_size , filter_size , out_channels , in_channels ]

        with tf.variable_scope( name ):
            weights_init = tf.get_variable( 'weight' , shape = weights_shape ,
                                        initializer = tf.contrib.layers.variance_scaling_initializer() ,
                                        dtype = tf.float32 )

        return weights_init

    ### INSTANCE NORM
    def instance_norm( self , net , name = None ):

        channels = net.get_shape().as_list()[3]
        var_shape = [ channels ]

        if self.type is None:
            mu , sq = tf.nn.moments( net , [ 1 , 2 ] )
            self.parsout.append( mu )
            self.parsout.append( sq )
        else:
#            n = int( net.shape[3] )
#            mu = self.parsin[ : , self.iter : self.iter + n ] ; self.iter += n
#            sq = self.parsin[ : , self.iter : self.iter + n ] ; self.iter += n
#            mu = approx( mu , 'approx' , self.iter   )
#            sq = approx( sq , 'approx' , self.iter+1 )
            mu , sq = tf.nn.moments( net , [ 1 , 2 ] )
            self.parsout.append( mu )
            self.parsout.append( sq )

        epsilon = 1e-3
        normalized = ( net - mu ) / ( sq + epsilon ) ** ( 0.5 )

        with tf.variable_scope( name ):
            shift = tf.get_variable( 'shift' , initializer = tf.zeros( var_shape ) , dtype = tf.float32 )
            scale = tf.get_variable( 'scale' , initializer = tf.ones(  var_shape ) , dtype = tf.float32 )

        return scale * normalized + shift

    ### REFLECT PADDING
    def reflect_padding( self , net , pad = 40 ):
        return tf.pad( net , [ [  0  ,  0  ] , [ pad , pad ] ,
                               [ pad , pad ] , [  0 ,   0 ] ] , "REFLECT" )

