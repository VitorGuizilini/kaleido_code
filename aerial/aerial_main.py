
import numpy as np
import kaleido as kld
from aerial_model import Model

######################

### COLORS
def colors( images ):
    actions = [ ( kld.img.convert , { 'map' : 'rgb2hsv' } ) ,
                ( kld.img.convert , { 'map' : 'rgb2lab' } ) ]
    nimages = images.copy()
    for act in actions:
        temp = kld.apply( images , act )
        for i in range( len( images ) ):
            nimages[i] = np.concatenate( [ nimages[i] , temp[i] ] , axis = 2 )
    return nimages

### AUGMENT
def augment( images , labels ):
    actions = [ ( kld.img.fliplr ) ,
                ( kld.img.fliptb ) ,
                ( kld.img.rotate , { 'angle' : 90  } ) ,
                ( kld.img.rotate , { 'angle' : 180 } ) ,
                ( kld.img.rotate , { 'angle' : 270 } ) ]
    nimages , nlabels = images.copy() , labels.copy()
    for act in actions:
        nimages += kld.apply( images , act )
        nlabels += kld.apply( labels , act )
    return nimages , nlabels

######################

parser = kld.mng.Parser( 'Aerial' )
parser.add_vrs_load_save_train_restart_store()
parser.add_num_epochs_eval_plot_every()

parser.add_str(  'path'    , d = 'aerial/cerrado' , h = 'Folder where information is stored' )
parser.add_str(  'dataset' , d = None             , h = 'Dataset to be used'                 )

parser.add_lint( 'max_data'    , d = [0,0] , h = 'Number of sampled Train/Valid points' , q = 2 )
parser.add_lint( 'batch_sizes' , d = [5,1] , h = 'Batch size for Train/Valid data'      , q = 2 )

parser.add_int(  'eval_idx' , d = 6     , h = 'Index to use for validation' )
parser.add_bol(  'augment'  , d = False , h = 'Augment training data'       )
parser.add_bol(  'colors'   , d = False , h = 'Include other colorspaces'   )

args = parser.args()

######################

path_data = '../../data/%s_%s' % ( args.path , args.dataset )
folders = kld.mng.Folder( path_data , recurse = 2 )
images , labels = folders.split( [ 'images' , 'labels' ] )

images_train , images_valid = images.split_files( args.eval_idx , [ '*.png' , '*.jpg' ] )
labels_train , labels_valid = labels.split_files( args.eval_idx , [ '*.png' , '*.jpg' ] )

[ images_train , labels_train ] = kld.lst.sample( [ images_train , labels_train ] , args.max_data[0] )
[ images_valid , labels_valid ] = kld.lst.sample( [ images_valid , labels_valid ] , args.max_data[1] )

[ images_train , images_valid ] = kld.img.load( [ images_train , images_valid ] , 'rgbn'  )
[ labels_train , labels_valid ] = kld.img.load( [ labels_train , labels_valid ] , 'grayn' )

if args.colors:  images_train , images_valid = colors( images_train ) , colors( images_valid )
if args.augment: images_train , labels_train = augment( images_train , labels_train )

######################

model = Model( [ images_train , labels_train ] ,
               [ images_valid , labels_valid ] , args )
if args.train: model.train()
else: model.test()
