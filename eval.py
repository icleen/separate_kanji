import os, sys, json
import numpy as np

from mwrapper import Wrapper

def main():
    config = sys.argv[1] if len(sys.argv) > 1 else 'configs/config.json'
    cont = sys.argv[2] if len(sys.argv) > 2 else 'cont'
    wrapper = Wrapper(config, cont=cont)
    wrapper.print_acc()

if __name__ == '__main__':
    main()
