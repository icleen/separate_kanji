import os, sys, json
import numpy as np

from mwrapper import Wrapper

def main():
    config = sys.argv[1] if len(sys.argv) > 1 else 'configs/config.json'
    wrapper = Wrapper(config)
    wrapper.train()

if __name__ == '__main__':
    main()
