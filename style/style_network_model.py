
import tensorflow as tf
import kaleido as kld

##### MODEL
class Model( kld.tf.model.baseA ):

######################################################

    ### __INIT__
    def __init__( self , data_content , data_style , data_valid , args ):

        self.resize_content = kld.init( kld.img.resize ,
                        size = ( args.sizes[0] , args.sizes[0] ) , interp = 'bilinear' )
        resize_style = kld.init( kld.img.resize ,
                        size = ( args.sizes[1] , args.sizes[1] ) , interp = 'bilinear' )
        resize_valid = kld.init( kld.img.resize ,
                        size = args.sizes[2] , interp = 'bilinear' )

        data_style = kld.img.load( data_style , 'rgbn' , resize_style )
        data_valid = kld.img.load( data_valid , 'rgbn' , resize_valid )

        self.data_content = kld.mng.Batch( data_content , args.batch_size )
        self.data_style   = kld.mng.Batch( data_style , 1 )
        self.data_valid   = kld.mng.Batch( data_valid , 1 )

        path_logs = '../../logs/style/'
        model = '%s_%d-%d_net%s/' % ( args.style.split('/')[-1] , args.sizes[0] , args.sizes[1] , args.vrs )
        path_load = None if args.load is None else path_logs + model + args.load
        path_save = None if args.save is None else path_logs + model + args.save
        self.prepare( args , path_load , path_save )

######################################################

    ### BUILD
    def build( self , args ):

        content_layers_idx = {}
        kld.lst.Norm( args.wgt_content_layers )
        for layer , weight in zip( args.content_layers , args.wgt_content_layers ):
            content_layers_idx[ layer ] = weight

        style_layers_idx = {}
        kld.lst.Norm( args.wgt_style_layers )
        for layer , weight in zip( args.style_layers , args.wgt_style_layers ):
            style_layers_idx[ layer ] = weight

        vgg = kld.tf.arch.get( 'vgg19' )
        vrsnet   = kld.vrsmod( args , '/Network'   , args.vrs[0] , self.saver )
        vrsloss  = kld.vrsmod( args , '/Loss'      , args.vrs[1] , self.saver )
        vrsoptim = kld.vrsmod( args , '/Optimizer' , args.vrs[2] , self.saver )

        self.loss = vrsloss( vrsnet , vgg , content_layers_idx , style_layers_idx )
        self.optim = vrsoptim( self.loss.full , self.start_epoch , args.num_epochs ,
                                                self.data_content.num_batches() )

        self.input = kld.tf.plchf( [ None , None , None , 3 ] , 'input' )
        self.output  = vrsnet( 'Output' , self.input ).output

        self.loader.restore_scope( 'Network' )
        self.loader.restore_scope( 'Optimizer' )

######################################################

    ### SAVE LOOP
    def save_loop( self , losses , epoch ):
        if self.args.store:
            self.saver.scope( 'Network' )
            self.saver.scope( 'Optimizer' )
            self.saver.scalar( 'epoch' , epoch )
            self.saver.list( 'valid_losses' , [ epoch ] + losses )

    ### OPTIMIZE LOOP
    def optimize_loop( self , data_content , data_style , epoch ):

        data_content.reset( shuffle = True )
        for _ in self.loopEpoch( data_content , epoch ):
            content = data_content.next_batch()
            content = kld.img.load( content , 'rgbn' , self.resize_content )

            data_style.reset()
            for _ in data_style.range_size():
                style = data_style.next_batch( 1 )
                self.sess.run( self.optim.run ,
                         { self.loss.yc : content , self.loss.ys : style ,
                           self.optim.lrate : self.optim.LRate.next() } )

    ### EVALUATE LOOP
    def evaluate_loop( self , data_valid , data_style , epoch , caption , draw_flag ):

        losses = [ 0 ] * 4

        data_valid.reset()
        for i in self.loopEval( data_valid , caption ):
            valid = data_valid.next_batch()

            data_style.reset()
            for j in data_style.range_size():
                style = data_style.next_batch( 1 )

                loss = self.sess.run( self.loss.all ,
                                { self.loss.yc : valid , self.loss.ys : style } )
                kld.lst.Add( losses , loss )

                if draw_flag:
                    output = self.sess.run( self.output , { self.input : valid } )
                    for k in range( len( output ) ):
                        ik = i * data_valid.batch_size() + k
                        file = '%03d_%02d_%04d' % ( ik , j , epoch )
                        folder = 'evolution/%03d/%02d' % ( ik , j )
                        self.saver.image( self.phase , output[k] , file , folder )

        return kld.lst.div( losses , data_valid.num_batches() * data_style.num_batches() )

######################################################

    ### DRAW FIRST
    def draw_first( self , data_valid , data_style ):
        for i in data_valid.range_size():
            for j in data_style.range_size():
                file = '%03d_%02d' % ( i , j )
                folder = 'evolution/%03d/%02d' % ( i , j )
                self.saver.image( self.phase , data_valid[i] , file , folder )

    ### DISPLAY
    def display( self , name = ' ' * 5 , epoch = 0 , losses = [ 0 ] * 4 ):
        str =  kld.dsp.count( name.upper() , epoch , self.args.num_epochs )
        return '| {} | Content: {:<1.7e} | Style: {:<1.7e} | TotVar: {:<1.7e} | Full: {:<1.7e} |'.format(
                    str , losses[0] , losses[1] , losses[2] , losses[3] )

######################################################

    ### DRAW
    def draw( self , data_valid , data_style ):
        self.draw_first( data_valid , data_style )

    ### EVALUATE
    def evaluate( self , data_valid , data_style , epoch ):
        kld.dsp.print_hline( self.width )
        losses_valid = self.evaluate_loop( data_valid , data_style , epoch , 'valid' , True )
        kld.dsp.print_hline( self.width )
        print( self.display( 'valid' , epoch , losses_valid ) )
        kld.dsp.print_hline( self.width )
        self.save_loop( losses_valid , epoch )

######################################################

    ### TRAIN
    def train( self ):

        self.test()
        for epoch in range( self.start_epoch + 1 , self.args.num_epochs + 1 ):
            self.optimize_loop( self.data_content , self.data_style , epoch )
            if kld.chk.iter_to( epoch , self.args.eval_every , self.args.num_epochs ):
                self.evaluate( self.data_valid , self.data_style , epoch )

    ### TEST
    def test( self ):

        self.draw( self.data_valid , self.data_style )
        self.evaluate( self.data_valid , self.data_style , self.start_epoch )

######################################################
