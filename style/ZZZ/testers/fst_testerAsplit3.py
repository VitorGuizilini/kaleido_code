
import copy
import importlib
import tensorflow as tf
import kaleido as kld
import numpy as np
import scipy
from fst_Algorithm import *
from tqdm import tqdm

### PARS DICT
def pars_dict( parsout , parsin ):
    pars_dict = {}
    pars_dict[ parsin ] = parsout
    return pars_dict

### LOAD
def load( file , resize1 , resize2 ):
    input = kld.img.load( file , 'rgbn' , resize1 )
    small = kld.img.load( file , 'rgbn' , resize2 )
    return input , small


##### TESTER
class Tester( Algorithm ):

    ### __INIT__
    def __init__( self , args ):
        self.preInit()

        self.args = args
        self.sess = tf.Session()
        self.load_network( 2 )

        self.build( args )
        self.test( args )

    ### BUILD
    def build( self , args ):
        self.preBuild()

        path = '../../logs/fast_style_transfer/%s' % ( args.model_dir )
        self.Saver = kld.log.Saver( path , self.sess )
        self.Saver.start_image( 'results' )

        self.x0 = kld.tf.plchf( [ None , None , 3 ] , 'input0' )
        self.xi0 = tf.expand_dims( self.x0 , 0 )
        self.x1 = kld.tf.plchf( [ None , None , 3 ] , 'input1' )
        self.xi1 = tf.expand_dims( self.x1 , 0 )

        with tf.variable_scope( '' , reuse = tf.AUTO_REUSE ):
            self.out0 , _ , self.pars0out = args.net[0].build( self.xi0 )
            self.out1 , self.pars1in , self.pars1out = args.net[1].build( self.xi1 , self.pars0out.shape[1] )
            self.Saver.restore_model( 'network' )

#            extras = kld.tf.global_vars( 'approx' )
#            self.Saver.restore_model( 'instance' , extras )

#        with tf.variable_scope( 'training' ):
#            self.loss = tf.reduce_mean( tf.square( self.pars0out - self.pars1out ) )
#            self.optim = tf.train.AdamOptimizer( 1e-6 ).minimize( self.loss , var_list = extras )
#            extras = kld.tf.global_vars( "training" )
#            self.sess.run( kld.tf.init_op( extras ) )

    ### TEST
    def test( self , args ):
        self.preTest()

        size , pad = 256 , 32
        model_name = kld.pth.fname( args.model_dir )

        resize1 = kld.partial( kld.img.resize , size = 1024 , interp = 'bilinear' )
        resize2 = kld.partial( kld.img.resize , size = 1024 , interp = 'bilinear' )
        suffix = '%s_%d' % ( model_name , 1024 )

        path = '../../data/'
        files_train = kld.mng.Folder( path + 'coco2014' ).files( pat = '*.jpg' )
        file_test = kld.mng.Folder( path + 'pics' ).files( pat = '*.jpg' )[0]

        input_test , small_test = load( file_test , resize1 , resize2 )
        pars0test = self.sess.run( self.pars0out , { self.x0 : small_test } )
        file_name = kld.pth.name( file_test )

#        data_train = kld.mng.Batch( files_train )
        for i in range( 1 ):
#            data_train.reset( shuffle = True )
#            for j in tqdm( range( 2000 ) ):
#                file = data_train.next_batch()[0]
#                input , small = load( file , resize1 , resize2 )
#                pars0train = self.sess.run( self.pars0out , { self.x0 : small } )
#                self.sess.run( self.optim , { self.x0 : input , self.pars1in : pars0train } )
#            self.Saver.model( 'instance' )

#            print( i , self.sess.run( self.loss , { self.x0 : input_test , self.pars1in : pars0test } ) )
            output1 =  self.sess.run( self.out0 , { self.x0 : input_test } )
            output2 =  self.sess.run( self.out1 , { self.x1 : input_test , self.pars1in : pars0test } )
            path = 'full3_eval1_%s_%d_%s' % ( file_name , size , suffix )
            self.Saver.image( 'results' , output1 / 255 , path )
            path = 'full3_eval2_%s_%d_%s' % ( file_name , size , suffix )
            self.Saver.image( 'results' , output2 / 255 , path )




#        output1 = self.sess.run( self.out0 , { **{ self.x0 : small } } )
#        output2 = self.sess.run( self.out0 , { **{ self.x0 : input } } )
#        output3 = self.sess.run( self.out1 , { **{ self.x1 : input } , **pars_small } )
#        output4 = self.sess.run( self.out1 , { **{ self.x1 : input } , **pars_large } )
#        path = '%s/full3_eval1_%s_%d_%s' % ( file_dir , file_name , size , suffix )
#        kld.image.save( output1 , path )
#        path = '%s/full3_eval2_%s_%d_%s' % ( file_dir , file_name , size , suffix )
#        kld.image.save( output2 , path )
#        path = '%s/full3_eval3_%s_%d_%s' % ( file_dir , file_name , size , suffix )
#        kld.image.save( output3 , path )
#        path = '%s/full3_eval4_%s_%d_%s' % ( file_dir , file_name , size , suffix )
#        kld.image.save( output4 , path )





#            jmp = 16
#            for i in range( 0 , small.shape[0] - jmp , jmp ):
#                for j in range( 0 , small.shape[1] - jmp , jmp ):
#                    rndh = i*n#np.random.randint( 0 , h - jmp )
#                    rndw = j*n#np.random.randint( 0 , w - jmp )
#                    small[i:i+jmp,j:j+jmp] = input[ rndh:rndh+jmp ,
#                                                    rndw:rndw+jmp ]

#            for i in range( small.shape[0] ):
#                for j in range( small.shape[1] ):
#                    for k in range( small.shape[2] ):
#                        small[i,j,k] = input[ np.random.randint( 0 , h ) ,
#                                              np.random.randint( 0 , w ) , k ]

#            mean = np.mean( small , axis = ( 0 , 1 , 2 ) )
#            var  = np.var(  small , axis = ( 0 , 1 , 2 ) )
#            input = ( input - mean ) / ( ( var + 1e-6 ) )

#            all_pars3 = []

#            div = 0
#            hst , hfn , wst , wfn = 0 , 0 , 0 , 0
#            for i in range( 0 , n , 4 ):
#                for j in range( 0 , n , 4 ):

#                    hst , hfn = i * hs , ( i + 1 ) * hs
#                    wst , wfn = j * ws , ( j + 1 ) * ws
#                    hstp = 0 if i == 0     else - pad
#                    wstp = 0 if j == 0     else - pad
#                    hfnp = 0 if i == n - 1 else   pad
#                    wfnp = 0 if j == n - 1 else   pad

#                    input_ij = input[ hst + hstp : hfn + hfnp , wst + wstp : wfn + wfnp , : ]
#                    div += 1

#            div = 0
#            for i in range( n ):
#                for j in range( n ):

#                    div += 1
#                    hst , hfn = i * hs , ( i + 1 ) * hs
#                    wst , wfn = j * ws , ( j + 1 ) * ws
#                    input_ij = input[ hst : hfn , wst : wfn , : ]

#                    pars = self.sess.run( args.net.pars_calc , { self.x : input_ij } )
#                    if len( all_pars3 ) == 0:
#                        all_pars3 = pars
#                    else:
#                        for k in range( len( all_pars3 ) ):
#                            all_pars3[k] += pars[k]

#            for k in range( len( all_pars3 ) ):
#                all_pars3[k] /= div

#            all_pars1 = self.sess.run( self.calc1 , { self.x1 : small } )
#            all_pars2 = self.sess.run( self.calc2 , { self.x2 : input } )

#            pars_dict = {}
#            for k in range( len( all_pars1 ) ):
#                if k % 2 == 0: pars_dict[ args.net.pars_eval[k] ] = all_pars1[k]
#                if k % 2 == 1: pars_dict[ args.net.pars_eval[k] ] = all_pars1[k]








#            hst , hfn , wst , wfn = 0 , 0 , 0 , 0
#            for i in range( n ):
#                for j in range( n ):

#                    hst , hfn = i * hs , ( i + 1 ) * hs
#                    wst , wfn = j * ws , ( j + 1 ) * ws
#                    hstp = 0 if i == 0     else - pad
#                    wstp = 0 if j == 0     else - pad
#                    hfnp = 0 if i == n - 1 else   pad
#                    wfnp = 0 if j == n - 1 else   pad

#                    output_ij = self.sess.run( self.yh , { **{ self.x : input_ij } , **pars_dict } )
#                    canvasfull[ hst : hfn , wst : wfn , : ] = output_ij[ - hstp : hs - hstp , - wstp : ws - wstp , : ]
#                    canvas = np.zeros( input.shape )
#                    canvas[ hst + hstp : hfn + hfnp , wst + wstp : wfn + wfnp , : ] = output_ij
#                    path = '%s/split_%2d-%2d_%s_%s' % ( file_dir , i , j , file_name , suffix )
#                    kld.save_image( canvas , path )

#            path = '%s/fullsplit_%s_%s' % ( file_dir , file_name , suffix )
#            kld.save_image( canvasfull , path )


#        self.store_model( 'fast_style_transfer' )


