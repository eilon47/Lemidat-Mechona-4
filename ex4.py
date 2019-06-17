from parameters import *
from routines import epoch_routine, test_routine
import numpy as np
import copy
import torch
from gcommand_loader import GCommandLoader, GCommandTestLoader
from model import LeNet
import torch.optim as optim


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
    test_l = load_dataset(test, test_params[0], test_params[1])
    return train_l, valid_l, test_l


def get_model_and_optimizer(optimizer_type, lr, momentum, cuda):
    model = LeNet()
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
    model, optimizer = get_model_and_optimizer(OPTIMIZER, LEARNING_RATE, MOMENTUM, CUDA)
    best_model = None
    best_valid_loss = np.inf
    best_valid_acc = - np.inf
    for epoch in range(EPOCHS):
        valid_loss, valid_acc = epoch_routine(train, valid, model, optimizer, epoch, CUDA)
        if best_valid_loss >= valid_loss and valid_acc >= best_valid_acc:
            print("Found better model with loss {} and accuracy {}% on validation set".format(valid_loss, valid_acc))
            best_model = copy.deepcopy(model)
    test_routine(test, best_model, CUDA)

if __name__ == '__main__':
    main()





















