
import random
import collections
import numpy as np
import tensorflow as tf
import kaleido as kld

#####################
### NORMALIZE WEIGHTS
def normalize_weights( weights ):
    sum = 0
    for w in weights: sum += w
    for i in range( len( weights ) ): weights[i] /= sum

###################
### TRIM TRAIN DATA
def trim_train_data( data , batch_size ):
    mod = len( data ) % batch_size
    if mod > 0: data = data[ : -mod ]
    return data

#############
### GET SHAPE
def get_shape( t ):
    if len( t.get_shape().as_list() ) == 3:
        return [ tf.shape(t)[0] , tf.shape(t)[1] , tf.shape(t)[2] ]
    if len( t.get_shape().as_list() ) == 4:
        return [ tf.shape(t)[0] , tf.shape(t)[1] , tf.shape(t)[2] , tf.shape(t)[3] ]

###########
### TOFLOAT
def toFloat( t ):
    return tf.cast( t , tf.float32 )

#################
### ADD DIM
def add_dim( a , dim ):
    return np.expand_dims( a , dim )

#################
### REM DIM
def rem_dim( a , dim ):
    return np.squeeze( a , dim )

##############
### MAKE_SHAPE
def make_shape( size , rgb , scale ):
    size , channels = int( size ) , 3 if rgb else 1
    if size == 0 or scale: return [ None , None , channels ]
    else:                  return [ size , size , channels ]

#######################
### ORDERED SORTED DICT
def ordered_sorted_dict( d ):
    return collections.OrderedDict( sorted( d.items() ) )

################
### SHUFFLE LIST
def shuffle_list( list ):
    random.shuffle( list )






