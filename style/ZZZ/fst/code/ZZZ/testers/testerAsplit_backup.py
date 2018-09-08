
import copy
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

        with tf.variable_scope( '' , reuse = tf.AUTO_REUSE ):

            args.net.type = 'calc'
            args.net.build( self.xi )

            args.net.type = 'eval'
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
            size , pad = 256 , 32
            h , w , c = input.shape
            n = int( np.ceil( max( h , w ) / size ) )
            hs , ws = int( h / n ) , int( w / n )

            print( 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA' )

#            print( input.shape )

#            arr = []
#            i = 0
#            while i < input.shape[0]:
#                arr.append( i )
#                if ( i + 1 ) % 4 == 0 : i += 13
#                else: i += 1
#            arr = np.asarray( arr )
#            small = input[ arr , : , : ]

#            print( small.shape )

#            arr = []
#            i = 0
#            while i < input.shape[1]:
#                arr.append( i )
#                if ( i + 1 ) % 4 == 0 : i += 13
#                else: i += 1
#            arr = np.asarray( arr )
#            small = small[ : , arr , : ]

#            print( small.shape )

#            pars = self.sess.run( args.net.pars , feed_dict = { self.x : small } )

            import scipy
            small = scipy.misc.imresize( input , 1.0 / n , interp = 'nearest' )

#            arr = np.arange( 2 , input.shape[0] , 4 )
#            small = input[ arr , : , : ]
#            arr = np.arange( 2 , input.shape[1] , 4 )
#            small = small[ : , arr , : ]

            pars = self.sess.run( args.net.pars_calc , feed_dict = { self.x : small } )

#            for i in range( len( pars ) ):
#                if i % 2 == 1: pars[i] = np.sqrt( pars[i] * n * n )

#            h2 , w2 = int( h / 2 ) , int( w / 2 )
#            hs2 , ws2 = int( hs / 2 ) , int( ws / 2 )
#            pars = self.sess.run( args.net.pars , feed_dict = { self.x : input[ h2 - hs2 : h2 + hs2 ,
#                                                                                w2 - ws2 : w2 + ws2 , : ] } )

#            small = input[ hs:2*hs , ws:2*ws , : ]
#            pars = self.sess.run( args.net.pars , feed_dict = { self.x : small } )

            print( 'BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB' )

#            pars2 = self.sess.run( args.net.pars , feed_dict = { self.x : input } )

            print( 'CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC' )

#            pars3 = []
#            for i in range( n ):
#                for j in range( n ):
#                    input_ij = input[ i * hs : ( i + 1 ) * hs ,
#                                      j * ws : ( j + 1 ) * ws , : ]
#                    tmp = self.sess.run( args.net.pars , feed_dict = { self.x : input_ij } )
#                    if len( pars3 ) == 0:
#                        pars3 = tmp
##                        for k in range( len( pars3 ) ):
##                            if k % 2 == 1: pars3[k] = np.sqrt( pars3[k] )
#                    else:
#                        for k in range( len( pars3 ) ):
#                            if k % 2 == 0: pars3[k] += tmp[k]
#                            if k % 2 == 1: pars3[k] += tmp[k]
#            for k in range( len( pars3 ) ):
#                if k % 2 == 0: pars3[k] /= n ** 2
#                if k % 2 == 1: pars3[k] /= n ** 2

            print( 'DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD' )

#            pars = pars2
#            for i in range( len( pars ) ):
#                if i % 2 == 0: pars[i] = pars3[i]
#                if i % 2 == 1: pars[i] = pars3[i]

#            [ print( p.shape ) for p in pars ]

            print( len(pars) )
            print( len(args.net.pars_eval) )

            pars_dict = {}
            for i in range( len( pars ) ):
                pars_dict[args.net.pars_eval[i]] = pars[i]

#            for i in range( 1 , len(pars) , 2 ):
#                print( '###############################################' , i )
#                print( pars[i] / pars2[i] )

            print( 'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE' )

            canvasfull = np.zeros( input.shape )

            hst , hfn , wst , wfn = 0 , 0 , 0 , 0

#            for i in range( n ):

#                for j in range( n ):

#                    hst , hfn = i * hs , ( i + 1 ) * hs
#                    wst , wfn = j * ws , ( j + 1 ) * ws

#                    hstp = 0 if i == 0     else - pad
#                    wstp = 0 if j == 0     else - pad
#                    hfnp = 0 if i == n - 1 else   pad
#                    wfnp = 0 if j == n - 1 else   pad

#                    input_ij = input[ hst + hstp : hfn + hfnp , wst + wstp : wfn + wfnp , : ]
#                    output_ij = self.sess.run( self.yh , feed_dict = { **{ self.x : input_ij } , **pars_dict } )

#                    canvasfull[ hst : hfn , wst : wfn , : ] = output_ij[ - hstp : hs - hstp , - wstp : ws - wstp , : ]

#                    canvas = np.zeros( input.shape )
#                    canvas[ hst + hstp : hfn + hfnp , wst + wstp : wfn + wfnp , : ] = output_ij

#                    path = '%s/split_%2d-%2d_%s_%s' % ( file_dir , i , j , file_name , suffix )
#                    kld.save_image( canvas , path )

#            path = '%s/fullsplit_%s_%s' % ( file_dir , file_name , suffix )
#            kld.save_image( canvasfull , path )

            output = self.sess.run( self.yh , feed_dict = { **{ self.x : input } , **pars_dict } )

            path = '%s/full_%s_%s' % ( file_dir , file_name , suffix )
            kld.save_image( output , path )

#        self.store_model( 'fast_style_transfer' )


