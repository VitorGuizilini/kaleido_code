
import argparse
from testers.testerA import Tester
import kaleido as kld

### PARSE ARGS
def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument( '--input_dir'  , type = str )
    parser.add_argument( '--model_dir'  , type = str )
    parser.add_argument( '--network'    , type = str )

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
