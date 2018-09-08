
import importlib
import tensorflow as tf
import kaleido as kld
import numpy as np

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

        self.x = kld.plchf( [ None , None , 3 ] , 'input' )
        self.xi = tf.expand_dims( self.x , 0 )

        self.yh = args.net.build( self.xi )
        self.yh = tf.squeeze( self.yh )
        self.yh = tf.clip_by_value( self.yh , 0.0 , 255.0 )

    ### TEST
    def test( self , args ):
        self.preTest()

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

            size , pad = 256 , 128
            h , w , c = input.shape
            n = int( np.ceil( max( h , w ) / size ) )
            hs , ws = int( h / n ) , int( w / n )
            canvas = np.zeros( input.shape )
            for i in range( 0 , h , hs ):
                for j in range( 0 , w , ws ):

                    hst , hfn = i , i + hs
                    wst , wfn = j , j + ws

                    hstp , hfnp , wstp , wfnp = 0 , 0 , 0 , 0
                    if i > 0: hstp -= pad
                    if j > 0: wstp -= pad
                    if i < n - 1: hfnp += pad
                    if j < n - 1: wfnp += pad

                    input_ij = input[ hst + hstp : hfn + hfnp , wst + wstp : wfn + wfnp , : ]
                    output_ij = self.sess.run( self.yh , feed_dict = { self.x : input_ij } )
                    canvas[ hst : hfn , wst : wfn , : ] = output_ij[ - hstp : hs - hstp , - wstp : ws - wstp , : ]

            path = '%s/split_%s_%s' % ( file_dir , file_name , suffix )
            kld.save_image( canvas , path )

            output = self.sess.run( self.yh , feed_dict = { self.x : input } )
            path = '%s/%s_%s' % ( file_dir , file_name , suffix )
            kld.save_image( output , path )

        self.store_model( 'fast_style_transfer' )


