
import argparse
from testers.testerAsplit4 import Tester
import kaleido as kld

### PARSE ARGS
def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument( '--input_dir'  , type = str )
    parser.add_argument( '--model_dir'  , type = str )
    parser.add_argument( '--network'    , type = str )

    parser.add_argument( '--batch_size' , type = int )
    parser.add_argument( '--num_epochs' , type = int )
    parser.add_argument( '--learn_rate'  , nargs = '+' , type = str )

    parser.add_argument( '--image_content' , nargs = '+' , type = str )
    parser.add_argument( '--content_dir' , type = str )

    parser.add_argument( '--image_test' , nargs = '+' , type = str )

    return parser.parse_args()

### MAIN
def main():

    args = parse_args()
    if args is None: exit()
    print( '####### FAST STYLE TRANSFER #######' )
    Tester( args )

if __name__ == '__main__':
    main()
