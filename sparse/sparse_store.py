
import numpy as np
import kaleido as kld

parser = kld.manager.Parser()
parser.add_tfloat( 'scale' , [ 200 , 600 ] )
parser.add_string( 'folder' )
args = parser.args()

scale = 'size %s %s nearest' % ( args.scale[0] , args.scale[1] )
image_params = [ kld.image.Params( 'rgb'  , scale ) ,
                 kld.image.Params( 'mono' , scale ) ,
                 kld.image.Params( 'mono' , scale ) ]

path = '/media/vguizilini/ETERNIA/Datasets/kitti'
folders = kld.manager.Folder( path + '/' + args.folder )
folders.recurse( 1 )
folders.append( 'proc_kitti_nick' )

datasets = folders.separate()
typ , ext = [ 'imgs' , 'disp1' , 'disp2' ] , '*.png'

for dataset in datasets:

    name = dataset.path(0,1)
    files = dataset.files_from( typ , ext )
    pathi = '%s/%s_%sc/%s/%s/'% (  path , str( args.scale[0] ) ,
                                          str( args.scale[1] ) , args.folder , name )
    kld.aux.mkdir( pathi )

    print( '**** PROCESSING' , name , len( files[0] ) )

    imgs , disp1 , disp2 = files
    imgs  = kld.image.load( imgs  , image_params[0] )
    disp1 = kld.image.load( disp1 , image_params[1] )
    disp2 = kld.image.load( disp2 , image_params[2] )

    np.save( pathi + 'imgs.npy'  , imgs  )
    np.save( pathi + 'disp1.npy' , disp1 )
    np.save( pathi + 'disp2.npy' , disp2 )
    print( '**** DONE' , pathi )


