
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

        args.image_content = kld.prepare_image_dict( args.image_content )
        args.image_test    = kld.prepare_image_dict( args.image_test    )

        self.ycL = kld.get_dir_files( args.content_dir )
        args.num_iters = len( self.ycL ) // args.batch_size

        self.build( args )
        self.test( args )

    ### BUILD
    def build( self , args ):
        self.preBuild()

        self.x    = kld.plchf( [ None , None , None , 3 ] , 'input'  )
        self.xs1  = kld.plchf( [ None , None , None , 3 ] , 'small1' )
        self.xs2  = kld.plchf( [ None , None , None , 3 ] , 'small2' )

        with tf.variable_scope( '' , reuse = tf.AUTO_REUSE ):
            args.net.type = 'calc'
            self.pars1 = args.net.build( self.xs1 )
            args.net.type = 'eval'
            self.yh1 = args.net.build( self.x )
            self.yh1 = tf.squeeze( self.yh1 )
            self.yh1 = tf.clip_by_value( self.yh1 , 0.0 , 255.0 )
        vars_orig = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES , scope = '' )
        self.saver = tf.train.Saver()

        with tf.variable_scope( 'updt' , reuse = tf.AUTO_REUSE ):
            args.net.type = 'calc'
            self.pars2 = args.net.build( self.xs2 )
            args.net.type = 'eval'
            self.yh2 = args.net.build( self.x )
            self.yh2 = tf.squeeze( self.yh2 )
            self.yh2 = tf.clip_by_value( self.yh2 , 0.0 , 255.0 )
        vars_updt = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES , scope = 'updt' )

        self.op_updt = []
        for ( u , o ) in zip( vars_updt , vars_orig ):
            self.op_updt.append( tf.assign( u , o ) )

        self.loss = tf.reduce_mean( tf.square( self.yh2 - self.yh1 ) )
        grads_updt = tf.gradients( self.loss , vars_updt )

        self.lr = kld.plchf( None , 'learn_rate' )
        optimizer = tf.train.AdamOptimizer( self.lr )
        self.optim = optimizer.apply_gradients( zip( grads_updt , vars_updt ) )

    ### TEST
    def test( self , args ):
        self.preTest()

        size1 , size2 , pad = 1024 , 256 , 32
        model_name = kld.basename( args.model_dir )
        suffix = '%s_%d.jpg' % ( model_name , args.image_test['size'] )

        yc = np.zeros( [ args.batch_size ] + args.image_content['shape'] , dtype = np.float32 )

        yc0 = self.load_image( self.ycL[0] , args.image_content )
        hc , wc , cc = yc0.shape ;
        n1c = int( np.ceil( max( hc , wc ) / size1 ) )
        n2c = int( np.ceil( max( hc , wc ) / size2 ) )

        yc0 = self.load_image( self.ycL[0] , args.image_test )
        ht , wt , ct = yc0.shape ;
        n1t = int( np.ceil( max( ht , wt ) / size1 ) )
        n2t = int( np.ceil( max( ht , wt ) / size2 ) )

        self.load_model()
        self.sess.run( self.op_updt )
        for epoch in range( 0 , args.num_epochs ):

            kld.shuffle_list( self.ycL )
            lr = self.calc_learn_rate( epoch )

            files = kld.get_dir_files( args.input_dir )
            for file in files:

                print( '%d - %s' % ( args.image_test['size'] , file ) )

                file_name = kld.basename( file )[:-4]
                file_dir = '%s/%s' % ( args.input_dir , file_name )
                kld.make_dir( file_dir )

                input = self.load_image( file , args.image_test )

                small1 = scipy.misc.imresize( input , 1.0 / n1t , interp = 'nearest' )
                output1 = self.sess.run( self.yh1 , feed_dict = { self.x : [ input ] , self.xs1 : [ small1 ] } )
                path = '%s/full4A_%02d_%s_%s' % ( file_dir , epoch , file_name , suffix )
                kld.save_image( output1 , path )

                small2 = scipy.misc.imresize( input , 1.0 / n2t , interp = 'nearest' )
                output2 = self.sess.run( self.yh2 , feed_dict = { self.x : [ input ] , self.xs2 : [ small2 ] } )
                path = '%s/full4B_%02d_%s_%s' % ( file_dir , epoch , file_name , suffix )
                kld.save_image( output2 , path )

                print( input.shape )
                print( small1.shape )
                print( small2.shape )
                print( n1c , n2c , n1t , n2t )

            for iter in range( args.num_iters ):

                print( epoch , args.num_epochs , iter , args.num_iters )

                yc_small1 , yc_small2 = [] , []
                curr , last = self.next_idxs( iter )
                for j , path in enumerate( self.ycL[ curr:last ] ):
                    yc[j] = self.load_image( path , args.image_content )
                    yc_small1.append( scipy.misc.imresize( yc[j] , 1.0 / n1c , interp = 'nearest' ) )
                    yc_small2.append( scipy.misc.imresize( yc[j] , 1.0 / n2c , interp = 'nearest' ) )

                self.sess.run( [ self.optim ] ,
                        feed_dict = { self.x : yc ,
                                      self.xs1 : yc_small1 ,
                                      self.xs2 : yc_small2 , self.lr : lr } )




#        size1 , size2 , pad = 1024 , 256 , 32
#        model_name = kld.basename( args.model_dir )
#        suffix = '%s_%d.jpg' % ( model_name , args.image_test['size'] )

#        self.load_model()
#        self.sess.run( self.op_updt )
#        files = kld.get_dir_files( args.input_dir )
#        for file in files:

#            print( '%d - %s' % ( args.image_test['size'] , file ) )

#            file_name = kld.basename( file )[:-4]
#            file_dir = '%s/%s' % ( args.input_dir , file_name )
#            kld.make_dir( file_dir )

#            input = self.load_image( file , args.image_test )

#            h1 , w1 , c1 = input.shape ; n = int( np.ceil( max( h1 , w1 ) / size1 ) )
#            small1 = scipy.misc.imresize( input , 1.0 / n , interp = 'nearest' )

#            output1 = self.sess.run( self.yh1 , feed_dict = { self.x : input , self.xs : small1 } )
#            path = '%s/full4A_%s_%s' % ( file_dir , file_name , suffix )
#            kld.save_image( output1 , path )

#            h2 , w2 , c2 = input.shape ; n = int( np.ceil( max( h2 , w2 ) / size2 ) )
#            small2 = scipy.misc.imresize( input , 1.0 / n , interp = 'nearest' )

#            output2 = self.sess.run( self.yh2 , feed_dict = { self.x : input , self.xs : small2 } )
#            path = '%s/full4B_%s_%s' % ( file_dir , file_name , suffix )
#            kld.save_image( output2 , path )

#        self.store_model( 'fast_style_transfer' )


