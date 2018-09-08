




import tensorflow as tf
import kaleido as kld

pool = ( tf.nn.max_pool , { 'ksize'   : [ 1 , 3 , 3 , 1 ] ,
                            'strides' : [ 1 , 2 , 2 , 1 ] , 'padding' : 'SAME' } )

### CONV
def conv( input , name , channels , ksizes , strides , ops1 = None , ops2 = None ):
    convs = []
    for ksize in ksizes:
        conv = kld.tf.layer.conv2d( input , name + str( ksize ) , channels , ksize , strides )
        convs.append( conv )
    if strides == 2:
        conv = kld.tf.layer.conv2d( input , name + 'MP' , channels , 1 , 1 )
        convs.append( kld.apply( conv , pool ) )
    convs = tf.concat( convs , axis = 3 )
    convs = kld.apply( convs , ops1 )
    convs = kld.tf.layer.conv2d( convs , name + 'OUT' , channels , 1 , 1 )
    convs = kld.apply( convs , ops2 )
    return convs

### DECONV
def deconv( input , name , channels , ksizes , convs , ops1 = None , ops2 = None ):
    scale , deconvs = kld.tf.shape( convs ) , []
    for ksize in ksizes:
        deconv = kld.tf.layer.rsconv2d( input , name + str( ksize ) , channels , ksize , scale )
        deconvs.append( deconv )
    deconvs = tf.concat( deconvs , axis = 3 )
    deconvs = kld.apply( deconvs , ops1 )
    deconvs = tf.concat( [ deconvs , convs ] , axis = 3 )
    deconvs = kld.tf.layer.rsconv2d( deconvs , name + 'OUT' , channels , 1 , scale )
    deconvs = kld.apply( deconvs , ops2 )
    return deconvs

##### VRS
class vrs( kld.tf.vrs ):

    ### BUILD
    def build( self , data ):

        with tf.variable_scope( 'Placeholders' ):

            self.input = kld.tf.plchf( data.shape( 0 , with_none = True ) , 'input' )
            self.label = kld.tf.plchf( data.shape( 1 , with_none = True ) , 'label' )

            self.phase = kld.tf.plchb( None , 'phase'   )
            self.drop  = kld.tf.plchf( None , 'dropout' )
            self.dropval = 0.3

        with tf.variable_scope( 'Operations' ):

            wgts_init = ( tf.contrib.layers.xavier_initializer , { 'uniform' : False } )
            bias_init = tf.initializers.zeros

            kld.tf.layer.defaults( { 'wgts_init' : wgts_init , 'bias_init' : bias_init } )

            actv = kld.tf.ops.llrelu
            sigm = tf.nn.sigmoid

            norm = ( tf.contrib.layers.batch_norm ,
                                      { 'epsilon' : 1e-5 , 'center' : True , 'scale' : True ,
                                        'is_training' : self.phase } )
            drop = ( tf.nn.dropout ,
                        { 'keep_prob' : self.drop } )

            pack1 = [ actv ]
            pack2 = [ actv , norm ]

        with tf.variable_scope( 'Layers' ):

            input , label = tf.expand_dims( self.input , 3 ) , self.label
#            input = ( input - 0.5 ) * 2.0

            with tf.variable_scope( 'Encoder' ):

                conv1   =   conv(   input ,   'conv1' ,  32 , [5,3] ,   2   , pack1 , pack2 )
                conv2   =   conv(   conv1 ,   'conv2' ,  32 , [5,3] ,   2   , pack1 , pack2 )
                conv3   =   conv(   conv2 ,   'conv3' ,  64 ,   [3] ,   2   , pack1 , pack2 )
                conv4   =   conv(   conv3 ,   'conv4' , 128 ,   [3] ,   2   , pack1 , pack2 )
                latent  =   conv(   conv4 ,  'latent' , 256 ,   [3] ,   2   , pack1 , pack2 )

            with tf.variable_scope( 'Decoder' ):

                deconv4 = deconv(  latent , 'deconv4' , 128 ,   [3] , conv4 , pack1 , pack2 )
                deconv3 = deconv( deconv4 , 'deconv3' ,  64 ,   [3] , conv3 , pack1 , pack2 )
                deconv2 = deconv( deconv3 , 'deconv2' ,  32 ,   [3] , conv2 , pack1 , pack2 )
                deconv1 = deconv( deconv2 , 'deconv1' ,  32 , [5,3] , conv1 , pack1 , pack2 )
                logits  = deconv( deconv1 ,  'output' ,   1 , [5,3] , input , pack1 ,  []  )

            self.logits = logits = tf.squeeze( logits , axis = 3 )
            self.output = output = tf.nn.sigmoid( logits )






