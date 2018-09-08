
import tensorflow as tf
import kaleido as kld

##### MODEL
class Model( kld.tf.model.baseA ):

######################################################

    ### __INIT__
    def __init__( self , data_train , data_valid , args ):

        self.data_train = kld.mng.MultiBatch( data_train , args.batch_sizes[0] )
        self.data_valid = kld.mng.MultiBatch( data_valid , args.batch_sizes[1] )

        path_logs = '../../logs/%s/vrs%s/' % ( args.path , args.vrs )
        path_load = None if args.load is None else path_logs + args.load
        path_save = None if args.save is None else path_logs + args.save
        self.prepare( args , path_load , path_save )

        h , w , _ = self.data_train.shape(0)
        kld.plt.adjust( w = 20 , p = h / w / 1.5 )

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
    def save_loop( self , laprf_train , laprf_valid , epoch ):
        if self.args.store:
            self.saver.scope( 'Network' )
            self.saver.scope( 'Optimizer' )
            self.saver.scalar( 'epoch' , epoch )
            self.saver.list( 'train_laprf' , [ epoch ] + laprf_train )
            self.saver.list( 'valid_laprf' , [ epoch ] + laprf_valid )

    ### OPTIMIZE LOOP
    def optimize_loop( self , data , epoch ):

        data.reset( shuffle = True )
        for i in self.loopEpoch( data , epoch ):
            image , label = data.next_batch()
            self.sess.run( self.optim.run ,
                     { self.net.input : image , self.net.label : label ,
                       self.net.drop : self.net.dropval , self.net.phase : True ,
                       self.optim.lrate : self.optim.LRate.next() } )

    ### EVALUATE LOOP
    def evaluate_loop( self , data , epoch , caption , draw_flag ):

        laprf = [ 0 ] * 5

        data.reset()
        for i in self.loopEval( data , caption ):

            image , label = data.next_batch()
            output , loss = self.sess.run( [ self.net.output , self.loss.full ] ,
                                     { self.net.input : image , self.net.label : label ,
                                       self.net.drop : 1.0 , self.net.phase : False } )
            kld.lst.Add( laprf , [ loss ] + kld.stt.aprf( label , output ) )

            if draw_flag:
                for k in range( len( output ) ):
                    ik = i * data.batch_size() + k
                    file = '%03d_%04d_out' % ( ik , epoch )
                    folder = 'evolution/%s/%03d' % ( caption , ik )
                    self.saver.image( self.phase , output[k] , file , folder )
                    if kld.chk.iter_to( epoch , self.args.plot_every , self.args.num_epochs ):
                        file = '%04d_%03d' % ( epoch , ik )
                        folder = 'sequence/%s/%04d' % ( caption , epoch )
                        plt = self.prepare_plot( image[k] , label[k] , output[k] )
                        self.saver.image( self.phase , plt , file , folder )

        return kld.lst.div( laprf , data.num_batches() )

######################################################

    ### DRAW FIRST
    def draw_first( self , data , caption ):
        _ , label = data
        for i in data.range_size():
            file = '%03d_%04d_lbl' % ( i , 0 )
            folder = 'evolution/%s/%03d' % ( caption , i )
            self.saver.image( self.phase , label[i] , file , folder )

    ### DISPLAY
    def display( self , name = ' ' * 5 , epoch = 0 , laprf = [ 0 ] * 5 ):
        str =  kld.dsp.count( name.upper() , epoch , self.args.num_epochs )
        return '| {} | Loss: {:<1.7e} ' \
                  '| Acc.: {:<8.6f} | Prec.: {:<8.6f} ' \
                  '| Recl.: {:<8.6f} | F-Meas.: {:<8.6f} |'.format(
                    str , laprf[0] , laprf[1] , laprf[2] , laprf[3] , laprf[4] )

######################################################

    ### DRAW
    def draw( self , data_valid ):
        self.draw_first( data_valid , 'valid' )

    ### EVALUATE
    def evaluate( self , data_train , data_valid , epoch ):
        kld.dsp.print_hline( self.width )
        laprf_train = self.evaluate_loop( data_train , epoch , 'train' , False )
        laprf_valid = self.evaluate_loop( data_valid , epoch , 'valid' , True  )
        kld.dsp.print_hline( self.width )
        print( self.display( 'train' , epoch , laprf_train ) )
        print( self.display( 'valid' , epoch , laprf_valid ) )
        kld.dsp.print_hline( self.width )
        self.save_loop( laprf_train , laprf_valid , epoch )

######################################################

    ### TRAIN
    def train( self ):

        self.test()
        for epoch in range( self.start_epoch + 1 , self.args.num_epochs + 1 ):
            self.optimize_loop( self.data_train , epoch )
            if kld.chk.iter_to( epoch , self.args.eval_every , self.args.num_epochs ):
                self.evaluate( self.data_train , self.data_valid , epoch )

    ### TEST
    def test( self ):

        self.draw( self.data_valid )
        self.evaluate( self.data_train , self.data_valid , self.start_epoch )

######################################################

    ### PREPARE_PLOT
    def prepare_plot( self , image , label , output ):

        imlabel = image.copy();  imlabel[:,:,2] = label
        imoutput = image.copy(); imoutput[:,:,2] = output
        return kld.plt.block( 2 , 3 , [ image[:,:,:3] , label  , imlabel[:,:,:3] ,
                                        image[:,:,:3] , output , imoutput[:,:,:3] ] )

######################################################
