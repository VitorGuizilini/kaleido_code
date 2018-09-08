
import kaleido as kld
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize
import cv2

saver = kld.log.FreeSaver( 'actions' )

action1 = kld.init( kld.img.resize , size = ( 1024 , 1024 ) )
action2 = kld.init( kld.img.resize , size = (  128 ,  128 ) )

image1 = kld.img.load( 'defesa.jpg' , 'rgbn' , action1 )

#image2 = imresize( image1 , [  128 ,  128 ] )
#image3 = imresize( image2 , [ 1024 , 1024 ] )

image2 = cv2.resize( image1 , (  128 ,  128 ) , interpolation = cv2.INTER_CUBIC )
image3 = cv2.resize( image2 , ( 1024 , 1024 ) , interpolation = 0 )

#image2 = kld.apply( image1 , action2 )
#image3 = kld.apply( image2 , action1 )

saver.image( 'results' , image1 , 'image1' )
saver.image( 'results' , image3 , 'image3' )

#for i in range( 1000 ):
#    image2 = kld.apply( image , action2 )
#    saver.image( 'results' , image2 , 'act_images%d' % i )


