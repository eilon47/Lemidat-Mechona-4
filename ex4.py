from parameters import *
from routines import epoch_routine, test_routine
import numpy as np
import copy
import torch
from gcommand_loader import GCommandLoader, GCommandTestLoader
from model import NNModel
import torch.optim as optim
from google.colab import files


def load_dataset(fname, dataset_params, loader_params, is_test=False):
    if not is_test:
        dataset = GCommandLoader(fname, window_size=dataset_params["win_size"],
                                        window_stride=dataset_params["win_stride"],
                                        window_type=dataset_params["win_type"],
                                        normalize=dataset_params["normalize"])
    else:
        dataset = GCommandTestLoader(fname, window_size=dataset_params["win_size"],
                                            window_stride=dataset_params["win_stride"],
                                            window_type=dataset_params["win_type"],
                                            normalize=dataset_params["normalize"])

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=loader_params["batch_size"],
                                         shuffle=loader_params["shuffle"],
                                         num_workers=loader_params["num_of_workers"],
                                         pin_memory=loader_params["cuda"],
                                         sampler=loader_params["sampler"])
    return loader


def load_all(train, train_param, valid, valid_params, test, test_params):
    train_l = load_dataset(train, train_param[0], train_param[1])
    valid_l = load_dataset(valid, valid_params[0], valid_params[1])
    test_l = load_dataset(test, test_params[0], test_params[1], is_test=True)
    return train_l, valid_l, test_l


def get_model_and_optimizer(optimizer_type, lr, momentum, cuda):
    model = NNModel(in1=1, out1=32, in2=32, out2=32, kernel_size1=5, kernel_size2=7)
    if cuda:
        model = torch.nn.DataParallel(model).cuda()
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=momentum)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=momentum)
    return model, optimizer


def main():
    train, valid, test = load_all(TRAIN_PATH, [dataset_params, loader_params], VALID_PATH, [dataset_params, loader_params], TEST_PATH, [test_dataset_params, test_loader_params])
    fd = open("output.txt", 'w')
    for E in EPOCHS:
        for LR in LEARNING_RATE:
            for O in OPTIMIZER:
                for M in MOMENTUM:
                    for D in DROPOUT:
                        model, optimizer = get_model_and_optimizer(OPTIMIZER, LEARNING_RATE, MOMENTUM, CUDA)
                        best_model = None
                        best_valid_loss = np.inf
                        best_valid_acc = - np.inf
                        for epoch in range(EPOCHS):
                            valid_loss, valid_acc = epoch_routine(train, valid, model, optimizer, epoch, CUDA)
                            if best_valid_loss >= valid_loss and valid_acc >= best_valid_acc:
                                print("Found better model with loss {} and accuracy {}% on validation set".format(valid_loss, valid_acc))
                                best_model = copy.deepcopy(model)
                        fd.write("{}_{}_{}_{}_{}_{}\n".format(best_valid_acc, E, LR, O, M, D))
                        test_routine(test, best_model, CUDA, fname="{}_{}_{}_{}_{}_{}".format(best_valid_acc, E, LR, O, M, D))
                        files.download("{}_{}_{}_{}_{}_{}".format(best_valid_acc, E, LR, O, M, D))
    fd.close()
    files.download("output.txt")


if __name__ == '__main__':
    main()





















