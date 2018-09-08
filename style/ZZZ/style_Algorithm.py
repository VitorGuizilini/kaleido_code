
import time
import numpy as np
import tensorflow as tf
import kaleido as kld
from glob import glob 
import scipy

#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

###### ALGORITHM
#class Algorithm:

#    saver = None
#    prev_time , acc_time = time.time() , 0
#    training = True

#    ### PREINIT
#    def preInit( self ):
#        print( '#### STARTING ... %3.2fs' % ( time.time() - self.prev_time ) )
#        self.prev_time = time.time()

#    ### PREBUILD
#    def preBuild( self ):
#        print( '#### BUILDING ... %3.2fs' % ( time.time() - self.prev_time ) )
#        self.prev_time = time.time()

#    ### PRETRAIN
#    def preTrain( self ):
#        print( '#### SETUPING ... %3.2fs' % ( time.time() - self.prev_time ) )
#        self.prev_time = time.time()
#        self.training = True

#    ### PRETEST
#    def preTest( self ):
#        print( '#### SETUPING ... %3.2fs' % ( time.time() - self.prev_time ) )
#        self.prev_time = time.time()
#        self.training = False

#    ### PRESTART
#    def preStart( self ):
#        if self.training:
#              print( '#### TRAINING ... %3.2fs' % ( time.time() - self.prev_time ) )
#        else: print( '#### TESTING  ... %3.2fs' % ( time.time() - self.prev_time ) )
#        self.prev_time = time.time()

#    ### TRANSFORMS
#    def transform( self , x ): return x
#    def untransform( self , x ): return x

#    ### LOAD NETWORK
#    def load_network( self , n = 1 ):
#        Net = kld.pth.module( 'networks.fst_' + self.args.network , 'Network' )
#        self.args.net = [ Net( self.args ) for _ in range( n ) ]
#        if n == 1: self.args.net = self.args.net[0]

#    ### NEXT IDXS
#    def next_idxs( self , iter ):
#        curr = iter * self.args.batch_size
#        last = curr + self.args.batch_size
#        return curr , last

#    ### CALC LEARN RATE
#    def calc_learn_rate( self , epoch ):

#        if self.args.learn_rate[0] == 'fixed':

#            learn_rate = float( self.args.learn_rate[1] )
#            return learn_rate

#        if self.args.learn_rate[0] == 'linear':

#            epoch_start = float( self.args.learn_rate[1] )
#            start_rate  = float( self.args.learn_rate[2] )
#            finish_rate = float( self.args.learn_rate[3] )

#            if epoch_start >= 1: epoch_start = int( epoch_start )
#            else: epoch_start = int( epoch_start * self.args.num_epochs )

#            if epoch < epoch_start: return start_rate
#            else: return finish_rate + ( start_rate - finish_rate ) * \
#                             ( self.args.num_epochs - epoch       ) / \
#                             ( self.args.num_epochs - epoch_start )

#    ### TIME TO EVAL
#    def time_to_eval( self , iter ):
#        return ( iter + 1 ) % self.args.eval_every == 0 or \
#               ( iter + 1 ) == self.args.num_iters

#    ### PRINT COUNTERS
#    def print_counters( self , epoch , iter ):

#        pad_epochs = int( np.ceil( np.log10( self.args.num_epochs + 1 ) ) )
#        pad_iters  = int( np.ceil( np.log10( self.args.num_iters  + 1 ) ) )

#        print( '|| {epoch: >{pad_epochs}}/{num_epochs} '.format( \
#            epoch = epoch + 1 , pad_epochs = pad_epochs , num_epochs = self.args.num_epochs ) , end = '' )
#        print( '- {iter: >{pad_iters}}/{num_iters} '.format( \
#            iter  = iter + 1  , pad_iters  = pad_iters  , num_iters  = self.args.num_iters  ) , end = '' )

#    ### PRINT TIME
#    def print_time( self , epoch , iter ):

#        delta = time.time() - self.prev_time
#        tot_steps = self.args.num_epochs * self.args.num_iters
#        num_steps = epoch * self.args.num_iters + iter + 1

#        self.acc_time += delta
#        avg_time = self.acc_time / num_steps
#        tot_time = avg_time * tot_steps
#        self.prev_time = time.time()

#        print( '|| %02dh%02dm / %02dh%02dm ||' % ( self.acc_time // 3600 , self.acc_time % 3600 // 60 ,
#                                                        tot_time // 3600 ,      tot_time % 3600 // 60 ) )
#        self.values[-1] = self.values[-1] + [ self.acc_time ]

#################
#### LOAD TRAINER
#def load_trainer( args ):
#    trn = importlib.import_module( 'trainers.' + args.trainer )
#    trn = getattr( trn , 'Trainer' )
#    trn( args )
