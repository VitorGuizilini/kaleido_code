
import tensorflow as tf
import kaleido as kld

##### BASE
class base:

    ### CONV INIT VARS
    def conv_init_vars( self , input , out_channels , filter_size ,
                        transpose = False , name = None ):

        in_channels = kld.tf.shape( input )[3]

        if not transpose:
            weights_shape = [ filter_size , filter_size , in_channels , out_channels ]
        else:
            weights_shape = [ filter_size , filter_size , out_channels , in_channels ]

        with tf.variable_scope( name ):
            weights_init = tf.get_variable( 'weight' , shape = weights_shape ,
                                        initializer = tf.contrib.layers.variance_scaling_initializer() ,
                                        dtype = tf.float32 )

        return weights_init

    ### CONV LAYER
    def conv_layer( self , input , num_filters , filter_size , strides ,
                    padding = 'SAME' , relu = True , name = None ):

        weights_init = self.conv_init_vars( input , num_filters , filter_size , name = name )
        strides_shape = [ 1 , strides , strides , 1 ]

        input = tf.nn.conv2d( input , weights_init , strides_shape , padding = padding )
        input = self.instance_norm( input , name = name )
        if relu: input = tf.nn.relu( input )

        return input

    ### CONV TRANSPOSE LAYER
    def conv_transpose_layer( self , input , num_filters , filter_size , strides ,
                              padding = 'SAME' , relu = True , name = None ):

        weights_init = self.conv_init_vars( input , num_filters , filter_size , name = name , transpose = True )
        strides_shape = [ 1 , strides , strides , 1 ]

        batch , rows , cols , channels = kld.tf.shape( input )
        new_rows, new_cols = rows * strides , cols * strides
        new_shape = tf.stack( [ batch , new_rows , new_cols , num_filters ] )

        input = tf.nn.conv2d_transpose( input , weights_init , new_shape , strides_shape , padding = padding )
        input = self.instance_norm( input , name = name )
        if relu: input = tf.nn.relu( input )

        return input

    ### RESIDUAL BLOCK
    def residual_block( self , input , filter_size = 3 , name = None ):

        batch , rows , cols , channels = kld.tf.shape( input )
        conv1 = self.conv_layer( input , 128 , filter_size , 1 , padding = 'VALID' , relu = True  , name = name + '_1' )
        conv2 = self.conv_layer( conv1 , 128 , filter_size , 1 , padding = 'VALID' , relu = False , name = name + '_2' )

        added = tf.slice( input , [ 0 , 2 , 2 , 0 ] , [ batch , rows - 4 , cols - 4 , channels ] )
        resid = conv2 + added

        rows , cols = kld.tf.shape( resid )[1:3]
        return tf.reshape( resid , [ batch , rows , cols , channels ] )

    ### INSTANCE NORM
    def instance_norm( self , input , name = None ):

        channels = input.get_shape().as_list()[3]
        var_shape = [ channels ]

        if kld.chk.is_tsr( self.instance ):
            n = int( input.shape[3] )
            mu = tf.reshape( self.parsin[ self.iter : self.iter + n ] , [ -1 , n ] ); self.iter += n
            sq = tf.reshape( self.parsin[ self.iter : self.iter + n ] , [ -1 , n ] ); self.iter += n
        else:
            mu , sq = tf.nn.moments( input , [ 1 , 2 ] , keep_dims = True )
            if callable( self.instance ):
                instance = self.instance( input , '%s_%s' % ( name , self.iter ) )
                muinst , sqinst = instance.mu , instance.sq
            else: muinst , sqinst = mu , sq
            self.parsout.append( tf.reshape( muinst , [ -1 ] ) )
            self.parsout.append( tf.reshape( sqinst , [ -1 ] ) )

        epsilon = 1e-6
        normalized = ( input - mu ) / ( sq + epsilon ) ** ( 0.5 )

        with tf.variable_scope( name ):
            shift = tf.get_variable( 'shift' , initializer = tf.zeros( var_shape ) , dtype = tf.float32 )
            scale = tf.get_variable( 'scale' , initializer = tf.ones(  var_shape ) , dtype = tf.float32 )

        return scale * normalized + shift

    ### REFLECT PADDING
    def reflect_padding( self , input , pad = 40 ):
        return tf.pad( input , [ [  0  ,  0  ] , [ pad , pad ] ,
                                 [ pad , pad ] , [  0 ,   0 ] ] , "REFLECT" )

