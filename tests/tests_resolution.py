
import kaleido as kld
import matplotlib.pyplot as plt
import numpy as np

saver = kld.log.Saver( 'logs' , free = True )

resize = kld.partial( kld.img.resize , size = 0.5 )
image = kld.img.load( 'defesa.jpg' , 'rgbn' , resize )
print( image.shape )

h , w , c = image.shape
p = h / w

#kld.plt.start( 1 )

for i in range( 30 ):
    print( i )
    kld.plt.adjust( w = 3 , p = p )
    plt0 = kld.plt.block( 2 , 2 , [ image , image , image , image ] )
    saver.image( 'images' , plt0 , 'test0' )
    kld.plt.adjust( w = 5 , p = p )
    plt1 = kld.plt.block( 2 , 2 , [ image , image , image , np.square( image ) ] )
    saver.image( 'images' , plt1 , 'test1' )


