
import kaleido as kld

######################

parser = kld.mng.Parser( 'Image Breaker for Aerial' )

parser.add_str(  'path'   , d = 'aerial/cerrado' , h = 'Folder where information is stored'  )
parser.add_int(  'scale'  , d = 5                , h = 'Downsizing scale for original image' )
parser.add_lint( 'size'   , d = [ 128 , 128 ]    , h = 'Cropped image size' , q = 2 )
parser.add_str(  'interp' , d = 'nearest'        , h = 'Resize interpolation' )
parser.add_str(  'suffix' , d = None             , h = 'Dataset suffix' )

args = parser.args()

######################

path = '../../data/' + args.path
folders = kld.mng.Folder( path , recurse = 1 )
imgfiles , lblfiles = folders.split_files( [ 'images' , 'labels' ] , [ '*.jpg' ] )
sc , sx , sy = args.scale , args.size[0] , args.size[1]

def save_crop( image , x , y , type , saver , folder , cnt ):
    file = '{:03d}'.format( cnt )
    crop = image[ x : x + sx , y : y + sy ]
    saver.image( type , crop , file , folder )

hx , hy = sx // 2 , sy // 2
path = '{}_{}_{}_{}_{}'.format( path , sc , sx , sy , args.interp )
if args.suffix is not None: path += '_' + args.suffix
saver = kld.log.Saver( path , restart = True , free = True )
scale = kld.init( kld.img.resize , size = 1.0 / sc , interp = args.interp )

######################

for file in imgfiles:
    print( file )
    image = kld.img.load( file , 'rgbn'  , scale )
    cnt , folder = 0 , file.split('_')[1][:-4]
    for x in range( 0 , image.shape[0] - sx , hx ):
        for y in range( 0 , image.shape[1] - sy , hy ):
            save_crop( image , x , y , 'images' , saver , folder , cnt ) ; cnt += 1

for file in lblfiles:
    print( file )
    label = kld.img.load( file , 'grayn'  , scale )
    cnt , folder = 0 , file.split('_')[1][:-4]
    for x in range( 0 , label.shape[0] - sx , hx ):
        for y in range( 0 , label.shape[1] - sy , hy ):
            save_crop( label , x , y , 'labels' , saver , folder , cnt ) ; cnt += 1

saver.list( 'dims' , [ image.shape[0] // hx ,
                       image.shape[1] // hy ] )

######################
