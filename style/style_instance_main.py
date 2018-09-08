
import kaleido as kld
from style_instance_model import Model

######################

parser = kld.mng.Parser( 'Style Instance' )
parser.add_vrs_load_save_train_restart_store()
parser.add_num_epochs_eval_every()

parser.add_str( 'content' , d = 'coco2014'     , h = 'Content dataset'    )
parser.add_str( 'style'   , d = 'styles/udnie' , h = 'Style dataset'      )
parser.add_str( 'valid'   , d = 'pics'         , h = 'Validation dataset' )
parser.add_str( 'network' , d = None           , h = 'Network to be used' )

parser.add_int( 'max_content' , d = None )

parser.add_lint( 'sizes' , d = [1024,256] , h = 'Input/Small sizes' )
parser.add_int(  'pad'   , d = 32         , h = 'Padding' )

args = parser.args()

######################

path_data = '../../data/'
data_content = kld.mng.Folder( path_data + args.content ).files( pat = '*.jpg' , max = args.max_content )
data_valid   = kld.mng.Folder( path_data + args.valid   ).files( pat = '*.jpg' , max = args.max_content )

######################

model = Model( data_content , data_valid , args )
if args.train: model.train()
else: model.test()

