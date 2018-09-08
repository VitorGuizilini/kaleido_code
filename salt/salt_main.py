
import numpy as np
import kaleido as kld
from salt_model import Model

######################

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

parser.add_str(  'path' , d = 'salt' , h = 'Folder where information is stored' )

parser.add_int( 'max_data'   , d = 0  , h = 'Number of sampled Train points' )
parser.add_int( 'batch_size' , d = 50 , h = 'Batch size for Train data' )

parser.add_bol(  'augment'  , d = False , h = 'Augment training data'       )
parser.add_bol(  'colors'   , d = False , h = 'Include other colorspaces'   )

args = parser.args()

######################

path_data = '../../data/%s' % ( args.path )
folders = kld.mng.Folder( path_data , recurse = 1 )
images_str , labels_str , valids_str = folders.split( [ 'images' , 'masks' , 'valids' ] )

images_str = images_str.files( pat = '*.png' )
labels_str = labels_str.files( pat = '*.png' )
valids_str = valids_str.files( pat = '*.png' )

[ images_str , labels_str ] = kld.lst.sample( [ images_str , labels_str ] , args.max_data )

images = kld.img.load( images_str , 'grayn' )
labels = kld.img.load( labels_str , 'grayn' )
valids = kld.img.load( valids_str , 'grayn' )
valids_str = [ str.split('/')[-1][:-4] for str in valids_str ]

if args.augment: images , labels = augment( images , labels )

#######################

model = Model( [ images , labels ] , [ valids_str , valids ] , args )
if args.train: model.train()
else: model.test()
