
import copy
import importlib
import tensorflow as tf
import kaleido as kld
import numpy as np
import scipy

##### TESTER
class Tester( kld.Algorithm ):

    ### __INIT__
    def __init__( self , args ):
        self.preInit()

        self.args = args
        self.sess = tf.Session()
        self.load_network()

        args.image_test = kld.prepare_image_dict( args.image_test )

        self.build( args )
        self.test( args )

    ### BUILD
    def build( self , args ):
        self.preBuild()

        self.x   = kld.plchf( [ None , None , 3 ] , 'input' )
        self.xi  = tf.expand_dims( self.x , 0 )
        self.xs  = kld.plchf( [ None , None , 3 ] , 'small' )
        self.xsi = tf.expand_dims( self.xs , 0 )

        with tf.variable_scope( '' , reuse = tf.AUTO_REUSE ):
            args.net.type = 'calc'
            args.net.build( self.xsi )
            args.net.type = 'eval'
            self.yh = args.net.build( self.xi )
            self.yh = tf.squeeze( self.yh )
            self.yh = tf.clip_by_value( self.yh , 0.0 , 255.0 )

    ### TEST
    def test( self , args ):
        self.preTest()

        size , pad = 256 , 32
        model_name = kld.basename( args.model_dir )
        suffix = '%s_%d.jpg' % ( model_name , args.image_test['size'] )

        self.load_model()
        files = kld.get_dir_files( args.input_dir )
        for file in files:

            print( '%d - %s' % ( args.image_test['size'] , file ) )

            file_name = kld.basename( file )[:-4]
            file_dir = '%s/%s' % ( args.input_dir , file_name )
            kld.make_dir( file_dir )

            input = self.load_image( file , args.image_test )
            h , w , c = input.shape ; n = int( np.ceil( max( h , w ) / size ) )
#            small = scipy.misc.imresize( input , 1.0 / n , interp = 'nearest' )
            small = input

            output = self.sess.run( self.yh , feed_dict = { self.x : input , self.xs : small } )

            path = '%s/full_%s_%s' % ( file_dir , file_name , suffix )
            kld.save_image( output , path )

#        self.store_model( 'fast_style_transfer' )


