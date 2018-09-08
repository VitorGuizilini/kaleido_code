
import numpy as np
import tensorflow as tf
import kaleido as kld

##### MODEL
class Model( kld.tf.model.baseA ):

######################################################

    ### __INIT__
    def __init__( self , data_content , data_valid , args ):

        self.resize_input = kld.init( kld.img.resize , size = args.sizes[0] , interp = 'bilinear' )
        self.resize_small = kld.init( kld.img.resize , size = args.sizes[1] , interp = 'bilinear' )

        data_valid = kld.img.load( data_valid , 'rgbn' , self.resize_input )

        self.data_content = kld.mng.Batch( data_content , 1 )
        self.data_valid   = kld.mng.Batch( data_valid   , 1 )

        path_logs = '../../logs/style/'
        model = '%s_%d-%d_inst%s/' % ( args.style.split('/')[-1] , args.sizes[0] , args.sizes[1] , args.vrs )
        if args.network is not None: path_load = path_logs + args.network
        else: path_load = None if args.load is None else path_logs + model + args.load
        path_save = None if args.save is None else path_logs + model + args.save
        self.prepare( args , path_load , path_save )

######################################################

    ### BUILD
    def build( self , args ):

        self.x0 = kld.tf.plchf( [ None , None , None , 3 ] , 'input0' )
        self.x1 = kld.tf.plchf( [ None , None , None , 3 ] , 'input1' )
        self.x2 = kld.tf.plchf( [ None , None , None , 3 ] , 'input2' )

        vrsnet   = kld.vrsmod( args , '/Network'   , args.vrs[0] , self.saver )
        vrsinst  = kld.vrsmod( args , '/Instance'  , args.vrs[1] , self.saver )
        vrsoptim = kld.vrsmod( args , '/Optimizer' , args.vrs[2] , self.saver )

        net0 = vrsnet( self.x0 )
        self.out0 , self.pars0out = net0.output , net0.parsout

        net1 = vrsnet( self.x1 , vrsinst )
        self.pars1out = net1.parsout

        net2 = vrsnet( self.x2 , self.pars0out )
        self.out2 , self.pars2in  = net2.output , net2.parsin

        self.loss = tf.reduce_mean( tf.square( self.pars0out - self.pars1out ) )

        self.optim = vrsoptim( self.loss , self.start_epoch , args.num_epochs ,
                                           self.data_content.num_batches() , 'Instance' )

        self.loader.restore_scope( 'Network' )
        self.loader.restore_scope( 'Instance' )
        self.loader.restore_scope( 'Optimizer' )

        kld.tf.nodes_to_freeze( [ self.x1       , 'inp1'     ] , [ self.x2      , 'inp2'    ] ,
                                [ self.pars1out , 'pars1out' ] , [ self.pars2in , 'pars2in' ] ,
                                [ self.out2     , 'out2'     ] )
        self.saver.model( 'android' )

######################################################

    ### SAVE LOOP
    def save_loop( self , losses , epoch ):
        if self.args.store:
            self.saver.scope( 'Network' )
            self.saver.scope( 'Instance' )
            self.saver.scope( 'Optimizer' )
            self.saver.scalar( 'epoch' , epoch )
            self.saver.list( 'valid_losses' , [ epoch ] + losses )

    ### OPTIMIZE LOOP
    def optimize_loop( self , data , epoch ):

        data.reset( shuffle = True )
        for _ in self.loopEpoch( data , epoch ):
            content = data.next_batch()
            content = kld.img.load( content , 'rgbn' , self.resize_input )
            small = kld.apply( content , self.resize_small )
            self.sess.run( self.optim.run , { self.x0 : content , self.x1 : small ,
                                              self.optim.lrate : self.optim.LRate.next() } )

    ### EVALUATE LOOP
    def evaluate_loop( self , data , epoch , caption , draw_flag ):

        losses = 0

        data.reset()
        for i in self.loopEval( data , caption ):
            input = data.next_batch()
            small = kld.apply( input , self.resize_small )

            losses += self.sess.run( self.loss , { self.x0 : input , self.x1 : small } )

            if draw_flag:
                for k in range( len( input ) ):
                    ik = i * data.batch_size() + k
                    file = '%03d_%04d' % ( ik , epoch )
                    folder = 'evolution/%03d' % ( ik )

                    pars0test = self.sess.run( self.pars0out , { self.x0 : small } )
                    pars1test = self.sess.run( self.pars1out , { self.x1 : small } )

                    out0  = self.sess.run( self.out0 , { self.x0 : input } )[0]
                    out20 = self.sess.run( self.out2 , { self.x2 : input , self.pars2in : pars0test } )[0]
                    out21 = self.sess.run( self.out2 , { self.x2 : input , self.pars2in : pars1test } )[0]

                    full0 , full20 , full21 = self.reconstruct( input[0] , small[0] , pars0test , pars1test )

                    self.saver.image( self.phase , out0   , file + '_out0'   , folder )
                    self.saver.image( self.phase , out20  , file + '_out20'  , folder )
                    self.saver.image( self.phase , out21  , file + '_out21'  , folder )
                    self.saver.image( self.phase , full0  , file + '_full0'  , folder )
                    self.saver.image( self.phase , full20 , file + '_full20' , folder )
                    self.saver.image( self.phase , full21 , file + '_full21' , folder )

        return losses / data.num_batches()

######################################################

    ### DRAW FIRST
    def draw_first( self , data_valid ):
        for i in data_valid.range_size():
            valid = data_valid.next_batch( 1 )[0]
            file = '%03d' % ( i )
            folder = 'evolution/%03d' % ( i )
            self.saver.image( self.phase , valid , file + '_raw' , folder )

    ### DISPLAY
    def display( self , name = ' ' , epoch = 0 , losses = 0 ):
        str =  kld.dsp.count( name.upper() , epoch , self.args.num_epochs )
        return '| {} | Losses: {:<1.7e} |'.format( str , losses )

######################################################

    ### DRAW
    def draw( self , data_valid ):
        self.draw_first( data_valid )

    ### EVALUATE
    def evaluate( self , data_valid , epoch ):
        kld.dsp.print_hline( self.width )
        losses_valid = self.evaluate_loop( data_valid , epoch , 'valid' , True )
        kld.dsp.print_hline( self.width )
        print( self.display( 'valid' , epoch , losses_valid ) )
        kld.dsp.print_hline( self.width )
        self.save_loop( losses_valid , epoch )

######################################################

    ### TRAIN
    def train( self ):

        self.test()
        for epoch in range( self.start_epoch + 1 , self.args.num_epochs + 1 ):
            self.optimize_loop( self.data_content , epoch )
            if kld.chk.iter_to( epoch , self.args.eval_every , self.args.num_epochs ):
                self.evaluate( self.data_valid , epoch )

    ### TEST
    def test( self ):

        self.draw( self.data_valid )
        self.evaluate( self.data_valid , self.start_epoch )

######################################################

    ### RECONSTRUCT
    def reconstruct( self , input , small , pars0 = None , pars1 = None ):

        sx , sy = input.shape[:2]
        hx , hy = small.shape[:2]
        nx , ny = sx // hx , sy // hy

        full0 = np.zeros( input.shape , dtype = np.float32 )
        full20 = np.zeros( input.shape , dtype = np.float32 )
        full21 = np.zeros( input.shape , dtype = np.float32 )

        for i in range( nx ):
            for j in range( ny ):

                stx , fnx = ( i ) * hx , ( i + 1 ) * hx
                sty , fny = ( j ) * hy , ( j + 1 ) * hy

                stxp = 0 if i == 0      else - self.args.pad
                styp = 0 if j == 0      else - self.args.pad
                fnxp = 0 if i == nx - 1 else   self.args.pad
                fnyp = 0 if j == ny - 1 else   self.args.pad

                inputij = input[ stx + stxp : fnx + fnxp , sty + styp : fny + fnyp ]

                blk0 = self.sess.run( self.out0 , { self.x0 : [ inputij ] } )[0]
                full0[ stx:fnx , sty:fny ] = blk0[ - stxp : - stxp + hx , - styp : - styp + hy ]

                if pars0 is not None:
                    blk20 = self.sess.run( self.out2 , { self.x2 : [ inputij ] , self.pars2in : pars0 } )[0]
                    full20[ stx:fnx , sty:fny ] = blk20[ - stxp : - stxp + hx , - styp : - styp + hy ]

                if pars1 is not None:
                    blk21 = self.sess.run( self.out2 , { self.x2 : [ inputij ] , self.pars2in : pars1 } )[0]
                    full21[ stx:fnx , sty:fny ] = blk21[ - stxp : - stxp + hx , - styp : - styp + hy ]

        return full0 , full20 , full21

######################################################

