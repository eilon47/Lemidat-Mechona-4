from __future__ import print_function
import sys
import torch.nn.functional as F
from torch.autograd import Variable
from os import path
import time

def print_progress_bar (iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total:
        print()


def epoch_routine(train_loader, valid_loader, model, optimizer, epoch, cuda):
    start = time.time()
    avg_loss_epoch = train_routine(train_loader, model, optimizer, epoch, cuda)
    print("Average loss for epoch {} is {}".format(epoch, avg_loss_epoch))
    valid_loss, valid_acc = validate_routine(valid_loader, model, epoch, cuda)
    print('\nValidation set: Avg loss: {:.4f}, Accuracy: ({:.0f}%)\n'.format(valid_loss, valid_acc))
    print("Total time for epoch {}".format(time.time()-start))
    return valid_loss, valid_acc


def train_routine(loader, model, optimizer, epoch, cuda):
    print("***********************************")
    print("***********  In Train  ************")
    print("***********  Epoch {}  ************".format(epoch))
    print("***********************************")
    model.train()
    total_loss = 0
    exmaples_so_far = 0
    total_examples = len(loader.dataset)
    for batch_index, (x, y) in enumerate(loader):
        if cuda:
            x, y = x.cuda(), y.cuda()
        x, y = Variable(x), Variable(y)
        optimizer.zero_grad()
        output = model(x)
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.data.item()
        exmaples_so_far += len(x)
        print_progress_bar(exmaples_so_far, len(loader.dataset), prefix="Progress in Epoch {}".format(epoch),
                         suffix="Complete")

        # if (exmaples_so_far) % 900 == 0:
        #     print('\nTrain::Epoch::{}:\t{} Examples of {} - ({:.0f}%)\tLoss: {:.6f}'.format(
        #                 epoch, exmaples_so_far, total_examples, 100.0* batch_index / total_examples, loss.item()))
    return float(total_loss / len(loader.dataset))


def validate_routine(loader, model, epoch ,cuda):
    print("***********************************")
    print("***********  In Valid  ************")
    print("***********  Epoch {}  ************".format(epoch))
    print("***********************************")
    model.eval()
    test_loss = 0
    correct = 0
    total_examples = len(loader.dataset)
    for data, target in loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data.item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= total_examples
    print("*************  Done  **************")
    return test_loss, float(100.0 * correct / total_examples)


def test_routine(loader, model, cuda, fname="test_y"):
    print("***********************************")
    print("************ In Test **************")
    print("***********************************")
    model.eval()
    fd = open(fname, 'w')
    line_format = "{}, {}\n"
    for data, full_path in loader:
        _, fname = path.split(full_path[0])
        if cuda:
            data = data.cuda()
        data = Variable(data)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        # Convert Tensor to int
        pred = int(pred[0,0].tolist())
        fd.write(line_format.format(fname, str(pred)))
    fd.close()
    print("*************  Done  **************")

