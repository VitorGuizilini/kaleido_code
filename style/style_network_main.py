
import numpy as np
import kaleido as kld
from style_network_model import Model

######################

parser = kld.mng.Parser( 'Style Network' )
parser.add_vrs_load_save_train_restart_store()
parser.add_num_epochs_eval_every()

parser.add_str( 'content' , d = 'coco2014'     , h = 'Content dataset'    )
parser.add_str( 'style'   , d = 'styles/udnie' , h = 'Style dataset'      )
parser.add_str( 'valid'   , d = 'pics'         , h = 'Validation dataset' )

parser.add_int( 'batch_size'  , d = 1    )
parser.add_int( 'max_content' , d = None )

parser.add_lint( 'sizes' , d = [512,512,1024] , h = 'Content/Style/Validation sizes' )

parser.add_lstr( 'style_layers'       , d = [ 'relu1_1' , 'relu2_1' , 'relu3_1' , 'relu4_1' , 'relu5_1' ] )
parser.add_lflt( 'wgt_style_layers'   , d = [    1.0    ,    1.0    ,    1.0    ,    1.0    ,    1.0    ] )
parser.add_lstr( 'content_layers'     , d = [ 'relu4_2' ] )
parser.add_lflt( 'wgt_content_layers' , d = [    1.0    ] )

parser.add_flt( 'wgt_content' , d = 7.5e+0 , h = 'Weight for content'         )
parser.add_flt( 'wgt_style'   , d = 5.0e+2 , h = 'Weight for style'           )
parser.add_flt( 'wgt_totvar'  , d = 2.0e+2 , h = 'Weight for total variation' )

args = parser.args()

######################

path_data = '../../data/'
data_content = kld.mng.Folder( path_data + args.content ).files( pat = '*.jpg' , max = args.max_content )
data_style   = kld.mng.Folder( path_data + args.style   ).files( pat = '*.jpg' )
data_valid   = kld.mng.Folder( path_data + args.valid   ).files( pat = '*.jpg' , max = args.max_content )

######################

model = Model( data_content , data_style , data_valid , args )
if args.train: model.train()
else: model.test()
