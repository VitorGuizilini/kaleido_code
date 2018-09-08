
import tensorflow as tf
import kaleido as kld

##### MODEL
class Model( kld.tf.model.baseA ):

######################################################

    ### __INIT__
    def __init__( self , data_train , data_valid , args ):

        self.data_train = kld.mng.MultiBatch( data_train , args.batch_size )
        self.data_valid = kld.mng.MultiBatch( data_valid , 1 )

        path_logs = '../../logs/%s/vrs%s/' % ( args.path , args.vrs )
        path_load = None if args.load is None else path_logs + args.load
        path_save = None if args.save is None else path_logs + args.save
        self.prepare( args , path_load , path_save )

        h , w = self.data_train.shape(0)
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
    def save_loop( self , laprf , epoch ):
        if self.args.store:
            self.saver.scope( 'Network' )
            self.saver.scope( 'Optimizer' )
            self.saver.scalar( 'epoch' , epoch )
            self.saver.list( 'laprf' , [ epoch ] + laprf )

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

        return kld.lst.div( laprf , data.num_batches() )

######################################################

    ### DISPLAY
    def display( self , name = ' ' * 5 , epoch = 0 , laprf = [ 0 ] * 5 ):
        str =  kld.dsp.count( name.upper() , epoch , self.args.num_epochs )
        return '| {} | Loss: {:<1.7e} ' \
                  '| Acc.: {:<8.6f} | Prec.: {:<8.6f} ' \
                  '| Recl.: {:<8.6f} | F-Meas.: {:<8.6f} |'.format(
                    str , laprf[0] , laprf[1] , laprf[2] , laprf[3] , laprf[4] )

######################################################

    ### EVALUATE
    def evaluate( self , data_train , data_valid , epoch ):
        kld.dsp.print_hline( self.width )
        laprf = self.evaluate_loop( data_train , epoch , 'train' , False )
        kld.dsp.print_hline( self.width )
        print( self.display( 'train' , epoch , laprf ) )
        kld.dsp.print_hline( self.width )
        self.save_loop( laprf , epoch )

######################################################

    ### TRAIN
    def train( self ):

        self.test()
        for epoch in range( self.start_epoch + 1 , self.args.num_epochs + 1 ):
            self.optimize_loop( self.data_train , epoch )
            if kld.chk.iter_to( epoch , self.args.eval_every , self.args.num_epochs ):
                self.evaluate( self.data_train , self.data_valid , epoch )
        self.submission( self.data_valid , self.args.num_epochs )


    ### TEST
    def test( self ):

        self.evaluate( self.data_train , self.data_valid , self.start_epoch )

######################################################

    ### SUBMISSION
    def submission( self , data , epoch ):

        f = self.saver.new_file( 'submission_%05d.cvs' % epoch , 'cvs' )

        f.write( 'id,rle_mask\n' )
        for i in data.range_batches():
            str , image = data.next_batch()
            output = self.sess.run( self.net.output ,
                                     { self.net.input : image , self.net.drop : 1.0 ,
                                       self.net.phase : False } )

            for j in kld.rlen( str ):

                f.write( str[j] + ',' )

                start = 0
                for c in range( output[j].shape[1] ):
                    for r in range( output[j].shape[0] ):
                        if start == 0:
                            if output[j][r,c] > 0.5:
                                f.write( '%d ' % ( output[j].shape[0] * c + r + 1 ) )
                                start = 1
                        else:
                            if output[j][r,c] > 0.5:
                                start += 1
                            else:
                                f.write( '%d ' % ( start ) )
                                start = 0

                if start > 0:
                    f.write( '%d ' % ( start ) )

                f.write( '\n' )
        f.close()

######################################################
