
import numpy as np
import tensorflow as tf
import kaleido as kld

##### TRAINERBASE
class TrainerBase( kld.Algorithm ):

    ### __INIT__
    def __init__( self , args ):
        self.preInit()

        self.args = args
        self.sess = tf.Session()
        self.load_network()

        args.image_style   = kld.prepare_image_dict( args.image_style   )
        args.image_content = kld.prepare_image_dict( args.image_content )

        self.ysL = self.load_image( args.style_dir , args.image_style )
        self.ycL = kld.get_dir_files( args.content_dir )
        args.num_iters = len( self.ycL ) // args.batch_size

        self.style_layers   = kld.ordered_sorted_dict( args.style_layer_ids   )
        self.content_layers = kld.ordered_sorted_dict( args.content_layer_ids )

        self.build( args )
        self.train( args )

    ### TRAIN
    def train( self , args ):
        self.preTrain()

        trainable_variables = tf.trainable_variables()
        grads = tf.gradients( self.L_full , trainable_variables )

        self.lr = kld.plchf( None , 'learn_rate' )
        optimizer = tf.train.AdamOptimizer( self.lr )
        optim = optimizer.apply_gradients( zip( grads , trainable_variables ) )

        yc = np.zeros( [ args.batch_size ] + args.image_content['shape'] , dtype = np.float32 )

        epoch = self.load_model()
        for epoch in range( epoch , args.num_epochs ):

            kld.shuffle_list( self.ysL )
            kld.shuffle_list( self.ycL )
            lr = self.calc_learn_rate( epoch )

#            for iter in range( args.num_iters ):
            for iter in range( int( args.num_iters / 2 ) ):

                curr , last = self.next_idxs( iter )
                for j , path in enumerate( self.ycL[ curr:last ] ):
                    yc[j] = self.load_image( path , args.image_content )
                ys = self.ysL[ iter % len( self.ysL ) ]

                self.sess.run( [ optim ] ,
                        feed_dict = { self.yc1 : yc , self.yc2 : yc ,
                                      self.ys  : ys , self.lr  : lr } )

                if self.time_to_eval( iter ):
                    L_full , L_style , L_content , L_totvar = self.sess.run(
                        [ self.L_full , self.L_style , self.L_content , self.L_totvar ] ,
                        feed_dict = { self.yc1 : yc , self.yc2 : yc , self.ys  : ys } )
                    self.print_counters( epoch , iter )
                    print( '|| L_full : %3.5e | L_style : %3.5e | L_content : %3.5e | L_totvar : %3.5e' %
                            ( L_full , L_style , L_content , L_totvar ) , end = '' )
                    self.values.append( [ epoch , iter , L_full , L_style , L_content , L_totvar ] )
                    self.print_time( epoch , iter )

            self.save_model()

