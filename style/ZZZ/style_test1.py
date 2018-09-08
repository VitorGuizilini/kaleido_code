
import argparse
from testers.style_testerA import Tester
import kaleido as kld

### MAIN
def main():

    parser = kld.mng.Parser()
    parser.add_rstr( 'input_dir' )
    parser.add_rstr( 'model_dir' )
    parser.add_rstr( 'network'   )
    parser.add_rlint( 'sizes'    )

    args = parser.args()

    if args is None: exit()
    print( '####### FAST STYLE TRANSFER #######' )
    Tester( args )

if __name__ == '__main__':
    main()
