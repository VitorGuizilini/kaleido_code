
import argparse
import kaleido as kld
import misc.vgg19 as vgg

### PARSE ARGS
def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument( '--style_dir' , type = str )
    parser.add_argument( '--model_dir' , type = str )
    parser.add_argument( '--trainer'   , type = str )
    parser.add_argument( '--network'   , type = str )

    parser.add_argument( '--eval_every' , type = int )
    parser.add_argument( '--batch_size' , type = int )
    parser.add_argument( '--num_epochs' , type = int )

    parser.add_argument( '--image_style'   , nargs = '+' , type = str )
    parser.add_argument( '--image_content' , nargs = '+' , type = str )

    parser.add_argument( '--content_dir' , type = str )
    parser.add_argument( '--vgg_net'     , type = str )

    parser.add_argument( '--style_layers'       , nargs = '+' , type = str   )
    parser.add_argument( '--wgt_style_layers'   , nargs = '+' , type = float )
    parser.add_argument( '--content_layers'     , nargs = '+' , type = str   )
    parser.add_argument( '--wgt_content_layers' , nargs = '+' , type = float )

    parser.add_argument( '--learn_rate'  , nargs = '+' , type = str )
    parser.add_argument( '--wgt_style'   , type = float )
    parser.add_argument( '--wgt_content' , type = float )
    parser.add_argument( '--wgt_totvar'  , type = float )

    return parser.parse_args()

### MAIN
def main():

    args = parse_args()
    if args is None: exit()
    print( '####### FAST STYLE TRANSFER #######' )

    vgg_path = kld.joinpath( args.vgg_net , vgg.MODEL_FILE_NAME )
    args.vgg_net = vgg.VGG19( vgg_path )

    args.style_layer_ids = {}
    kld.normalize_weights( args.wgt_style_layers )
    for layer , weight in zip( args.style_layers , args.wgt_style_layers ):
        args.style_layer_ids[ layer ] = weight

    args.content_layer_ids = {}
    kld.normalize_weights( args.wgt_content_layers )
    for layer , weight in zip( args.content_layers , args.wgt_content_layers ):
        args.content_layer_ids[ layer ] = weight

    kld.load_trainer( args )

if __name__ == '__main__':
    main()
