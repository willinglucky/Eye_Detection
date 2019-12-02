from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import sys
import matplotlib.pyplot as plt
import random
import numpy as np
#整体归一化
# normMean = [0.57458663, 0.5311794, 0.53112286]
# normStd = [0.28226474, 0.27916232, 0.278306]
# transforms.Normalize(normMean = [0.57458663, 0.5311794, 0.53112286], normStd = [0.28226474, 0.27916232, 0.278306])
# 眼部归一化
# normMean = [0.60297036, 0.5137168, 0.50260335]
# normStd = [0.24658357, 0.24121241, 0.24273053]
# transforms.Normalize(normMean = [0.60297036, 0.5137168, 0.50260335], normStd = [0.24658357, 0.24121241, 0.24273053])

# Writer will output to ./runs/ directory by default
# writer = SummaryWriter('./log')

seed = 60

torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
random.seed(seed)
np.random.seed(seed)
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for data in dataloders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).to(torch.float32)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

            if phase == 'train':
                train_acc_list.append(epoch_acc)
                train_loss_list.append(epoch_loss)
            else:
                test_acc_list.append(epoch_acc)
                test_loss_list.append(epoch_loss)
        scheduler.step()


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    x1 = range(0, num_epochs)
    plt.title('accuracy vs. epoches')
    plt.ylabel('accuracy')

    l1, = plt.plot(x1, train_acc_list,
                   label='train',
                   color='blue',
                    linewidth = 1.0,  # 线条宽度
                    linestyle = '-.' # 线条样式
                   )
    l2, = plt.plot(x1, test_acc_list,
                   color='red',  # 线条颜色
                   linewidth=1.0,  # 线条宽度
                   linestyle='-.',  # 线条样式
                   label='test'  # 标签
                   )

    # 使用ｌｅｇｅｎｄ绘制多条曲线
    plt.legend(handles=[l1, l2],
               labels=['train', 'test'],
               loc='best'
               )
    plt.savefig("acc.jpg")
    plt.clf()
    plt.close()

    l1, = plt.plot(x1, train_loss_list,
                   label='train',
                   color='blue',
                    linewidth = 1.0,  # 线条宽度
                    linestyle = '-.'  # 线条样式
                   )
    l2, = plt.plot(x1, test_loss_list,
                   color='red',  # 线条颜色
                   linewidth=1.0,  # 线条宽度
                   linestyle='-.',  # 线条样式
                   label='test'  # 标签
                   )

    # 使用ｌｅｇｅｎｄ绘制多条曲线
    plt.legend(handles=[l1, l2],
               labels=['train', 'test'],
               loc='best'
               )
    plt.xlabel('Loss vs. epoches')
    plt.ylabel('loss')
    plt.savefig("loss.jpg")
    plt.clf()
    plt.close()
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def test_model(model, criterion):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0


    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0.0

    # Iterate over data.
    for data in dataloders['test']:
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data).to(torch.float32)

    epoch_loss = running_loss / dataset_sizes['test']
    epoch_acc = running_corrects / dataset_sizes['test']

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        'test', epoch_loss, epoch_acc))

    time_elapsed = time.time() - since
    print('Testing complete in ',time_elapsed)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
if __name__ == '__main__':

    # data_transform, pay attention that the input of Normalize() is Tensor and the input of RandomResizedCrop() or RandomHorizontalFlip() is PIL Image
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.60297036, 0.5137168, 0.50260335],[0.24658357, 0.24121241, 0.24273053])
            #imgnet归一化
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.60297036, 0.5137168, 0.50260335],[0.24658357, 0.24121241, 0.24273053])
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # your image data file
    data_dir = './eye_segmentation_dataset'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x]) for x in ['train', 'test']}
    # wrap your data and label into Tensor
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=4,
                                                 shuffle=True,
                                                 num_workers=4) for x in ['train', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    # use gpu or not
    use_gpu = torch.cuda.is_available()
    print(use_gpu)
    # get model and replace the original fc layer with your fc layer
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)

    if use_gpu:
        model_ft = model_ft.cuda()

    # define loss function
    criterion = nn.CrossEntropyLoss()

    if sys.argv[1] == 'train':
        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0005, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

        model_ft = train_model(model=model_ft,
                               criterion=criterion,
                               optimizer=optimizer_ft,
                               scheduler=exp_lr_scheduler,
                               num_epochs=60)
        torch.save(model_ft.state_dict(), './params.pth')
    elif sys.argv[1] == 'test':
        model_ft.load_state_dict(torch.load('params_v1.0.pth'))
        test_model(model=model_ft,criterion=criterion)