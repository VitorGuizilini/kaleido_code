
import numpy as np
import kaleido as kld
from sparse_model import Model

######################

parser = kld.mng.Parser( 'Sparse' )
parser.add_vrs_load_save_train_restart_store()
parser.add_num_epochs_eval_plot_every()

parser.add_str(  'path'    , d = 'sparse'  , h = 'Folder where information is stored' )
parser.add_str(  'dataset' , d = '200_600' , h = 'Dataset to be used'                 )

parser.add_lint( 'max_data'    , d = [0,0,20] , h = 'Number of sampled Train/Valid/Plots points' , q = 3 )
parser.add_lint( 'batch_sizes' , d = [1,1,1]  , h = 'Batch size for Train/Valid/Plots data'      , q = 3 )

parser.add_str(  'sample_type'  , d = 'prob'  , h = 'Type of sampling to be applied' , c = ['prob','area'] )
parser.add_lint( 'sample_train' , d = [ 100 ] , h = 'Sampling interval for training'  )
parser.add_lint( 'sample_eval'  , d = [ 100 ] , h = 'Sampling values to be evaluated' )

args = parser.args()

######################

path_data = '../../data/kitti'
#path_data = '/media/vguizilini/ETERNIA/Datasets/kitti'
path_data = '%s/%s' % ( path_data , args.dataset )
folders = kld.mng.Folder( path_data , recurse = 2 )
train , valid = folders.split( 3 )

image_train = [ np.load( folder + 'imgs.npy'  ) for folder in train ]
disp1_train = [ np.load( folder + 'disp1.npy' ) for folder in train ]
disp2_train = [ np.load( folder + 'disp2.npy' ) for folder in train ]
image_train = kld.cvt.npy2lst( image_train )
disp1_train = kld.cvt.npy2lst( disp1_train )
disp2_train = kld.cvt.npy2lst( disp2_train )

image_valid = [ np.load( folder + 'imgs.npy'  ) for folder in valid ]
disp1_valid = [ np.load( folder + 'disp1.npy' ) for folder in valid ]
disp2_valid = [ np.load( folder + 'disp2.npy' ) for folder in valid ]
image_valid = kld.cvt.npy2lst( image_valid )
disp1_valid = kld.cvt.npy2lst( disp1_valid )
disp2_valid = kld.cvt.npy2lst( disp2_valid )

files_train = [ image_train , disp1_train , disp2_train ]
files_valid = [ image_valid , disp1_valid , disp2_valid ]

files_train = kld.lst.sample( files_train , args.max_data[0] )
files_valid = kld.lst.sample( files_valid , args.max_data[1] )
files_plots = kld.lst.sample( files_valid , args.max_data[2] )

######################

model = Model( files_train , files_valid , files_plots , args )
if args.train: model.train()
else: model.test()

