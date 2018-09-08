
import argparse
from testers.style_testerB import Tester
import kaleido as kld

### MAIN
def main():

    parser = kld.mng.Parser()

    parser.add_rstr(  'input_dir' , 'results' )
    parser.add_rstr(  'model_dir' , 'udnie_512-512_BBE' )
    parser.add_rstr(  'network'   )
    parser.add_rlint( 'sizes'     , [ 1024 , 256 ] )
    parser.add_bool(  'train'     , False )

    args = parser.args()

    if args is None: exit()
    print( '####### FAST STYLE TRANSFER #######' )
    Tester( args )

if __name__ == '__main__':
    main()
