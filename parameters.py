import os
import platform
import torch

#  OS
IS_WIN = platform.system().lower() == "windows"

#  FILES
BASEDIR = "samples_data"
TRAIN_PATH = os.path.join(".", BASEDIR, "train")
VALID_PATH = os.path.join(".", BASEDIR, "valid")
TEST_PATH = os.path.join(".", BASEDIR, "test")

#  CUDA
CUDA_RUN = True
CUDA = CUDA_RUN and torch.cuda.is_available()
SEED = 1234

#  LOADER
BATCH_SIZE = 100
BATCH_SIZE_FOR_TEST = 1
NUM_OF_WORKERS = 20

#  DATA SET
WINDOW_TYPE = "hamming"
WINDOW_SIZE = .02
NORMALIZE = True
WINDOW_STRIDE = .01


# TRAIN ROUTINE
EPOCHS = [20, 30, 50, 100]
LEARNING_RATE = [0.001, 0.002, 0.005]
MOMENTUM = [0.1, 0.5, 0.9]
OPTIMIZER = ["adam", "sgd"]
DROPOUT = [0.1, 0.3, 0.4]

loader_params = {
    "batch_size": BATCH_SIZE,
    "shuffle":True,
    "sampler":None,
    "cuda":CUDA,
    "num_of_workers":NUM_OF_WORKERS
}
dataset_params = {
    "win_size":WINDOW_SIZE,
    "win_stride":WINDOW_STRIDE,
    "win_type":WINDOW_TYPE,
    "normalize":NORMALIZE
}
test_loader_params = {
    "batch_size": BATCH_SIZE_FOR_TEST,
    "shuffle":None,
    "sampler":None,
    "cuda":CUDA,
    "num_of_workers":NUM_OF_WORKERS
}
test_dataset_params = {
    "win_size":WINDOW_SIZE,
    "win_stride":WINDOW_STRIDE,
    "win_type":WINDOW_TYPE,
    "normalize":NORMALIZE
}

