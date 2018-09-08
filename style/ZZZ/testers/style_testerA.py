
import tensorflow as tf
import kaleido as kld
import numpy as np
import scipy
from tqdm import tqdm

### LOAD
def load( file , resize1 , resize2 ):
    input = kld.img.load( file , 'rgbn' , resize1 )
    small = kld.apply( input , resize2 )
    return input , small

##### TESTER
class Tester():

    ### __INIT__
    def __init__( self , args ):

        self.args = args
        self.sess = tf.Session()
        self.load_network( 3 )

        self.build( args )
        self.test( args )

    ### LOAD NETWORK
    def load_network( self , n = 1 ):
        Net = kld.pth.module( 'networks.style_' + self.args.network , 'Network' )
        self.args.net = [ Net( self.args ) for _ in range( n ) ]
        if n == 1: self.args.net = self.args.net[0]

    ### BUILD
    def build( self , args ):

        path = '../../logs/fast_style_transfer/%s' % ( args.model_dir )
        self.Saver = kld.log.Saver( path , self.sess )

        self.x = kld.tf.plchf( [ None , None , 3 ] , 'input0' )
        self.xi = tf.expand_dims( self.x , 0 )

        with tf.variable_scope( '' , reuse = tf.AUTO_REUSE ):
            self.out0 ,       _      , self.pars0out = args.net[0].build( self.xi )
            self.out1 , self.pars1in , self.pars1out = args.net[1].build( self.xi , self.pars0out , False )
            self.Saver.restore_model( 'network' )
            self.out2 , self.pars2in , self.pars2out = args.net[2].build( self.xi , self.pars0out , True )
            extras = kld.tf.global_vars( 'approx' )
            self.Saver.restore_model( 'instance' , extras )
        with tf.variable_scope( 'training' ):
            self.loss = tf.reduce_mean( tf.square( self.pars0out - self.pars2out ) )
            self.optim = tf.train.AdamOptimizer( 1e-5 ).minimize( self.loss , var_list = extras )
            extras = kld.tf.global_vars( "training" )
            self.sess.run( kld.tf.init_op( extras ) )

    ### TEST
    def test( self , args ):

        size , pad = 256 , 32
        model_name = kld.pth.fname( args.model_dir )

        resize1 = kld.partial( kld.img.resize , size = args.sizes[0] , interp = 'bilinear' )
        resize2 = kld.partial( kld.img.resize , size = args.sizes[1] , interp = 'bilinear' )
        suffix = '%s_%d' % ( model_name , 1024 )

        path = '../../data/'
        files_train = kld.mng.Folder( path + 'coco2014' ).files( pat = '*.jpg' )
#        files_train = kld.mng.Folder( path + 'pics' ).files( pat = '*.jpg' )
        file_test = kld.mng.Folder( path + 'pics' ).files( pat = '*.jpg' )[0]

        input_test , small_test = load( file_test , resize1 , resize2 )
        pars0test = self.sess.run( self.pars0out , { self.x : small_test } )
        file_name = kld.pth.name( file_test )

        data_train = kld.mng.Batch( files_train )
        for i in range( 2000 ):
#            data_train.reset( shuffle = True )
#            for j in tqdm( range( 1000 ) ):
#                timer = kld.mng.Timer()
#                file = data_train.next_batch()[0]
#                input , small = load( file , resize1 , resize2 )
#                pars0train = self.sess.run( self.pars0out , { self.x : small } )
#                self.sess.run( self.optim , { self.x : input , self.pars2in : pars0train } )
#            self.Saver.model( 'instance' )

            print( i , self.sess.run( self.loss , { self.x : input_test , self.pars2in : pars0test } ) )

            output0 =  self.sess.run( self.out0 , { self.x : input_test } )
            output1 =  self.sess.run( self.out1 , { self.x : input_test , self.pars1in : pars0test } )
            output2 =  self.sess.run( self.out2 , { self.x : input_test , self.pars2in : pars0test } )
            path = 'full3_eval0_%s_%d_%s' % ( file_name , size , suffix )
            self.Saver.image( 'results' , output0 , path )
            path = 'full3_eval1_%s_%d_%s' % ( file_name , size , suffix )
            self.Saver.image( 'results' , output1 , path )
            path = 'full3_eval2_%s_%d_%s' % ( file_name , size , suffix )
            self.Saver.image( 'results' , output2 , path )

