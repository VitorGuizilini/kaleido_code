
import os, argparse
import tensorflow as tf
import kaleido as kld

model = 'udnie_1024-256_instABA'
path = '../../logs/style/%s/android01/models/android' % ( model )
name1 = "./%s.pb" % ( model )
name2 = "./%s.cry" % ( model )

nodes , sess = kld.tf.freezecry( path , name2 )
print( nodes )

nodes , sess = kld.tf.unfreezecry( name2 )
print( nodes )














