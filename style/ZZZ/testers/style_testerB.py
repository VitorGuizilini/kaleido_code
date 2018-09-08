
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

    ### RECONSTRUCT
    def reconstruct( self , input , small , pars , pad ):

        sx , sy = input.shape[:2]
        hx , hy = small.shape[:2]
        nx , ny = int( sx / hx ) , int( sy / hy )

        full0 = np.zeros( input.shape , dtype = np.float32 )
        full2 = np.zeros( input.shape , dtype = np.float32 )

        for i in range( nx ):
            for j in range( ny ):

                stx , fnx = ( i ) * hx , ( i + 1 ) * hx
                sty , fny = ( j ) * hy , ( j + 1 ) * hy

                stxp = 0 if i == 0      else - pad
                styp = 0 if j == 0      else - pad
                fnxp = 0 if i == nx - 1 else   pad
                fnyp = 0 if j == ny - 1 else   pad

                inputij = input[ stx + stxp : fnx + fnxp , sty + styp : fny + fnyp ]

                blk0 = self.sess.run( self.out0 , { self.x : inputij } )
                blk2 = self.sess.run( self.out2 , { self.x : inputij , self.pars2in : pars } )
                full0[ stx:fnx , sty:fny ] = blk0[ - stxp : - stxp + hx , - styp : - styp + hy ]
                full2[ stx:fnx , sty:fny ] = blk2[ - stxp : - stxp + hx , - styp : - styp + hy ]

        return full0 , full2

    ### TEST
    def test( self , args ):

        pad , model_name = 32 , kld.pth.fname( args.model_dir )

        path = '../../data/'
        file_test = kld.mng.Folder( path + 'pics' ).files( pat = '*.jpg' )[0]
        file_name = kld.pth.name( file_test )

        resize1 = kld.partial( kld.img.resize , size = args.sizes[0] , interp = 'bilinear' )
        resize2 = kld.partial( kld.img.resize , size = args.sizes[1] , interp = 'bilinear' )

        input_test , small_test = load( file_test , resize1 , resize2 )
        pars0test = self.sess.run( self.pars0out , { self.x : small_test } )

        out0 =  self.sess.run( self.out0 , { self.x : input_test } )
        out1 =  self.sess.run( self.out1 , { self.x : input_test , self.pars1in : pars0test } )
        out2 =  self.sess.run( self.out2 , { self.x : input_test , self.pars2in : pars0test } )

        full0 , full2 = self.reconstruct( input_test , small_test , pars0test , pad )

        name = '_%s_%dx%d_%s' % ( file_name , args.sizes[0] , args.sizes[1] , model_name )
        self.Saver.image( 'results/images2' , out0  , 'split_eval0' + name )
        self.Saver.image( 'results/images2' , out1  , 'split_eval1' + name )
        self.Saver.image( 'results/images2' , out2  , 'split_eval2' + name )
        self.Saver.image( 'results/images2' , full0 , 'full_eval0'  + name )
        self.Saver.image( 'results/images2' , full2 , 'full_eval2'  + name )



