
import tensorflow as tf
import kaleido as kld

##### MODEL
class Model( kld.tf.model.baseA ):

######################################################

    ### __INIT__
    def __init__( self , data_train , data_valid , data_plots , args ):

        self.data_train = kld.mng.MultiBatch( data_train , args.batch_sizes[0] )
        self.data_valid = kld.mng.MultiBatch( data_valid , args.batch_sizes[1] )
        self.data_plots = kld.mng.MultiBatch( data_plots , args.batch_sizes[2] )

        path_logs = '../../logs/%s/vrs%s/' % ( args.path , args.vrs )
        path_load = None if args.load is None else path_logs + args.load
        path_save = None if args.save is None else path_logs + args.save
        if path_save is not None:
            path_save += '_' + args.sample_type[0] + str( int( args.sample_train[0] ) )
            if len( args.sample_train ) == 2: path_save += '-' + str( int( args.sample_train[1] ) )
        self.prepare( args , path_load , path_save )

        h , w , _ = self.data_train.shape(0)
        kld.plt.adjust( w = 25 , p = h / w )

######################################################

    ### BUILD
    def build( self , args ):

        vrsnet   = kld.vrsmod( args , '/Network'   , args.vrs[0] , self.saver )
        vrsloss  = kld.vrsmod( args , '/Loss'      , args.vrs[1] , self.saver )
        vrsoptim = kld.vrsmod( args , '/Optimizer' , args.vrs[2] , self.saver )

        self.net = vrsnet( self.data_train )
        self.loss = vrsloss( self.net )
        self.optim = vrsoptim( self.loss.full , self.start_epoch , args.num_epochs ,
                                                self.data_train.num_batches() )

        self.loader.restore_scope( 'Network'   )
        self.loader.restore_scope( 'Optimizer' )

######################################################

    ### SAVE LOOP
    def save_loop( self , losses_train , losses_valid , losses_plots , epoch ):
        if self.args.store:
            self.saver.scope( 'Network' )
            self.saver.scope( 'Optimizer' )
            self.saver.scalar( 'epoch' , epoch )
            self.saver.list( 'train_losses' , [ epoch ] + losses_train )
            self.saver.list( 'valid_losses' , [ epoch ] + losses_valid )
            self.saver.list( 'plots_losses' , [ epoch ] + losses_plots )

    ### OPTIMIZE LOOP
    def optimize_loop( self , data , epoch ):

        data.reset( shuffle = True  )
        for i in self.loopEpoch( data , epoch ):
            _ , disp1 , disp2 = data.next_batch()
            self.sess.run( self.optim.run ,
                     { self.net.input : self.sample( disp1 ) , self.net.label : disp2 ,
                       self.optim.lrate : self.optim.LRate.next() } )

    ### EVALUATE LOOP
    def evaluate_loop( self , data , epoch , caption , draw_flag ):

        losses = [ 0 ] * len( self.args.sample_eval )

        for j , smp in self.loopEval( self.args.sample_eval , caption , enum = True ):

            data.reset()
            for i in self.loopEval( data , '{} {:>3d}'.format( caption , smp ) , leave = False ):

                image , disp1 , disp2 = data.next_batch()
                sdisp1 = self.sample( disp1 , smp )
                output , loss = self.sess.run( [ self.net.output , self.loss.full ] ,
                                         { self.net.input : sdisp1 , self.net.label : disp2 } )
                losses[j] += loss

                if draw_flag:
                    for k in range( len( output ) ):
                        ik = i * data.batch_size() + k
                        file = '%03d_%03d_%04d_out' % ( smp , ik , epoch )
                        folder = 'evolution/%s/%03d/%03d' % ( caption , smp , ik )
                        self.saver.image( self.phase , output[k] , file , folder )
                        if kld.chk.iter_to( epoch , self.args.plot_every , self.args.num_epochs ):
                            file = '%03d_%04d_%03d_out' % ( smp , epoch , ik )
                            folder = 'sequence/%s/%03d/%04d' % ( caption , smp , epoch )
                            plt = self.prepare_plot( image[k] , sdisp1[k] , disp2[k] , output[k] )
                            self.saver.image( self.phase , plt , file , folder )

        return kld.lst.div( losses , data.num_batches() )

######################################################

    ### DRAW FIRST
    def draw_first( self , data , caption ):
        _ , disp1 , disp2 = data
        for smp in self.args.sample_eval:
            sdisp1 = self.sample( disp1 , smp )
            for j in data.range_size():
                sdisp1[j][ sdisp1[j] > 0 ] = 1.0
                file = '%03d_%03d_%04d_' % ( smp , j , 0 )
                folder = 'evolution/%s/%03d/%03d' % ( caption , smp , j )
                self.saver.image( self.phase , sdisp1[j] , file + 'inp' , folder )
                self.saver.image( self.phase ,  disp2[j] , file + 'lbl' , folder )

    ### DISPLAY
    def display( self , name = ' ' * 5 , epoch = 0 , losses = None ):
        if losses is None: losses = [ 0 ] * len( self.args.sample_eval )
        str =  '| ' + kld.dsp.count( name.upper() , epoch , self.args.num_epochs )
        for smp , loss in zip( self.args.sample_eval , losses ):
            str += ' | Sample {:d}: {:<1.5e}'.format( smp , loss )
        return str + ' |'

######################################################

    ### DRAW
    def draw( self , data_plots ):
        self.draw_first( data_plots , 'plots' )

    ### EVALUATE
    def evaluate( self , data_train , data_valid , data_plots , epoch ):
        kld.dsp.print_hline( self.width )
        losses_train = self.evaluate_loop( data_train , epoch , 'train' , False )
        losses_valid = self.evaluate_loop( data_valid , epoch , 'valid' , False )
        losses_plots = self.evaluate_loop( data_plots , epoch , 'plots' , True  )
        kld.dsp.print_hline( self.width )
        print( self.display( 'train' , epoch , losses_train ) )
        print( self.display( 'valid' , epoch , losses_valid ) )
        print( self.display( 'plots' , epoch , losses_plots ) )
        kld.dsp.print_hline( self.width )
        self.save_loop( losses_train , losses_valid , losses_plots , epoch )

######################################################

    ### TRAIN
    def train( self ):

        self.test()
        for epoch in range( self.start_epoch + 1 , self.args.num_epochs + 1 ):
            self.optimize_loop( self.data_train , epoch )
            if kld.chk.iter_to( epoch , self.args.eval_every , self.args.num_epochs ):
                self.evaluate( self.data_train , self.data_valid , self.data_plots , epoch )

    ### TEST
    def test( self ):

        self.draw( self.data_plots )
        self.evaluate( self.data_train , self.data_valid , self.data_plots , self.start_epoch )

######################################################

    ### SAMPLE
    def sample( self , image , rnd = None ):
        if self.args.sample_type == 'prob':
            if rnd is None:
                if len( self.args.sample_train ) == 1: rnd = self.args.sample_train[0]
                else: rnd = kld.rnd.f( self.args.sample_train[0] , self.args.sample_train[1] )
                return kld.img.sample_prob( image , rnd / 100 )
            else: return kld.img.sample_prob( image , rnd / 100 , 999 )
        elif self.args.sample_type == 'area':
            if rnd is None:
                if len( self.args.sample_train ) == 1: rnd = self.args.sample_train[0]
                else: rnd = kld.rnd.i( self.args.sample_train[0] , self.args.sample_train[1] )
                return kld.img.sample_area( image , rnd )
            else: return kld.img.sample_area( image , rnd , 999 )

    ### PREPARE PLOT
    def prepare_plot( self , image , disp1 , disp2 , output ):

        imdisp1 = image.copy()
        idx = disp1 > 0 ; imdisp1[idx,:] = 1.0
        return kld.plt.block( 2 , 2 , [ imdisp1 , disp1 , disp2 , output ] )

######################################################
