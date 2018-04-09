import tensorflow as tf
import numpy as np
import matplotlib as plt

def train():
    pass

def test():
    pass

# Main
if __name__ == "__main__":

    num_args = len(argv)
    if num_args > 1 and argv[1] == 'train':
        print("training model...")
        train()
    elif num_args > 1 and argv[1] == 'test':
        print("verifying model...")
        test()
    else:
        print("valid arguments: train or test")
