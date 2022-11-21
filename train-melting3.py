from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from pytorchtools import EarlyStopping
import torch.nn as nn
from torch.utils.data.dataset import *
from PIL import Image
from torch.nn import functional as F
import random
import shutil
from sklearn.model_selection import KFold, StratifiedKFold
import multiprocessing
from sklearn.metrics import confusion_matrix
import torch.utils.data as data

data_dir = os.getcwd()
# from[resnet, alexnet, inception,vgg]pick one pre-trained model
model= ["vgg",'resnet','alexnet']
num_classes = 3
batch_size = 8
num_epochs = 200
patience = 10
delta = 0.01
trial = 0
feature_extract =False # false== fine tuning
img_path = []
label = []  # 1 is solid, 0 is liquid
folder_path = ['test', 'train', 'val']
# Paths for image directory and model
EVAL_DIR = 'test'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
LOSS = []  # each trial
ACC = []


def train_model(model, dataloaders, criterion, optimizer, num_epochs, patience, delta, is_inception=False):
    since = time.time()

    val_acc_history = []
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {
        'train': {'loss': [], 'acc': []},
        'val': {'loss': [], 'acc': []}
    }
    early_stopping = EarlyStopping(patience=patience, verbose=True, delta=delta)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # write_log('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            if phase == 'val':
                valid_loss = epoch_loss

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # write_log('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            history[phase]['loss'].append(epoch_loss)
            history[phase]['acc'].append(epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()
        # early_stopping needs the validation loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            # write_log("Early stopping")
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    write_log('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # write_log('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(torch.load('checkpoint.pt'))
    return model, val_acc_history, history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
 """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
 """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
 """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    #
    #    elif model_name == "squeezenet":
    #        """ Squeezenet
    # """
    #        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
    #        set_parameter_requires_grad(model_ft, feature_extract)
    #        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    #        model_ft.num_classes = num_classes
    #        input_size = 224
    #
    #    elif model_name == "densenet":
    #        """ Densenet
    # """
    #        model_ft = models.densenet121(pretrained=use_pretrained)
    #        set_parameter_requires_grad(model_ft, feature_extract)
    #        num_ftrs = model_ft.classifier.in_features
    #        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
    #        input_size = 224

    elif model_name == "inception":
        """ Inception v3
 Be careful, expects (299,299) sized images and has auxiliary output
 """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # 处理辅助网络
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # 处理主要网络
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def image_pre():
    global label, img_path, folder_path
    # empty test and train folder to load split images
    label = []
    img_path = []
    for path in folder_path:
        for labeldir in os.listdir(path):
            label_path = str(path + '/' + labeldir)
            shutil.rmtree(label_path)
            os.makedirs(label_path)

    data = open('kfold.txt', 'r')
    data_lines = data.readlines()
    # label lists solid or liquid
    # image_path lists image name
    for data_line in data_lines:
        data = data_line.strip().split(' ')
        img_path.append(data[0])
        if data[1] == 'solid':
            label.append(2)
        elif data[1] =='half':
            label.append(1)
        else:
            label.append(0)




def empty_imagefolder():
    global folder_path
    # empty test and train folder to load split images
    for path in folder_path:
        for labeldir in os.listdir(path):
            label_path = str(path + '/' + labeldir)
            shutil.rmtree(label_path)
            os.makedirs(label_path)


def data_load():
    global data_transforms, dataloaders_dict, image_datasets
    # Data augmentation and normalization for training
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")
    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloadersdr
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0,
                                       drop_last=True, pin_memory=True) for x in
        ['train', 'val']}


def scratch_train():
    # Initialize the non-pretrained version of the model used for this run
    scratch_model, _ = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False)
    scratch_model = scratch_model.to(device)
    scratch_optimizer = optim.SGD(scratch_model.parameters(), lr=0.001, momentum=0.9)
    scratch_criterion = nn.CrossEntropyLoss()
    _, _, scratch_hist = train_model(scratch_model, dataloaders_dict, scratch_criterion, scratch_optimizer,
                                     num_epochs=num_epochs, patience=patience, delta=delta,
                                     is_inception=(model_name == "inception"))

    # Plot the training curves of validation accuracy vs. number
    #  of training epochs for the transfer learning method and
    #  the model trained from scratch
    ohist = []
    shist = []

    ohist = [h.cpu().numpy() for h in hist]
    shist = [h.cpu().numpy() for h in scratch_hist]

    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1, num_epochs + 1), ohist, label="Pretrained val_acc")
    plt.plot(range(1, num_epochs + 1), shist, label="Scratch acc")
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, num_epochs + 1, 1.0))
    plt.legend()
    plt.show()


def plot_history(history):
    """Plot historical training and validation accuracies and losses

    Args:
        history (dict): training and validation losses and accuracies history.
                        {'train': {'loss': [], 'acc': []},
                         'val': {'loss': [], 'acc': []}}

    Returns:
        None
    """
    fig, ax1 = plt.subplots()

    # Correctly number epochs starting from 1
    epochs = np.arange(1, len(history['train']['loss']) + 1)

    # Plot losses
    ax1.plot(epochs, history['train']['loss'], 'g-')
    ax1.plot(epochs, history['val']['loss'], 'b-')
    ax1.set_xlabel('Training Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(['Training Loss', 'Validation Loss'], bbox_to_anchor=(0.6, 0.2))

    # find position of lowest validation loss
    minposs = history['val']['loss'].index(min(history['val']['loss'])) + 1

    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')
    plt.grid(True)
    # plt.legend()
    plt.tight_layout()

    # plt.show()    # plt.show()
    Path_name = model_name + ('FE' if feature_extract else 'FT')
    path_name = model_name + str(trial) + 'trial' + ('FE' if feature_extract else 'FT')
    plt.title(path_name)
    PATH = '{}.png'.format(path_name)
    PATH = os.path.join(Path_name, PATH)
    fig.savefig(PATH, bbox_inches='tight')

    # Plot accuracies
    # ax2 = ax1.twinx()
    # ax2.plot(epochs, history['train']['acc'], 'y-')
    # ax2.plot(epochs, history['val']['acc'], 'r')
    # ax2.set_ylabel('Accuracy')
    # ax2.legend(['Training Accuracy', 'Validation Accuracy'], bbox_to_anchor=(0.6, 0.8))

    # plt.legend(frameon=False)
    # plt.show()


def create_log_folder():  # create a new folder to save log file and info
    global Log_result, Path_name
    Path_name = model_name + ('FE' if feature_extract else 'FT')
    if not os.path.exists(Path_name):
        os.mkdir(Path_name)

    Log_result = (os.path.join(Path_name, 'log.txt'))


def write_log(meg):
    with open(Log_result, 'a') as f:
        f.write(meg + '\n')


def tranfer_file(train, test, val, label):
    for i in train:
        if label[i] == 2:
            shutil.copyfile(str('kfold/' + img_path[i]), 'train/solid/' + img_path[i])
        elif label[i] == 1:
            shutil.copyfile(str('kfold/' + img_path[i]), 'train/half/' + img_path[i])
        else:
            shutil.copyfile(str('kfold/' + img_path[i]), 'train/liquid/' + img_path[i])
    for i in test:
        if label[i] == 2:
            shutil.copyfile(str('kfold/' + img_path[i]), 'test/solid/' + img_path[i])
        elif label[i] == 1:
            shutil.copyfile(str('kfold/' + img_path[i]), 'test/half/' + img_path[i])
        else:
            shutil.copyfile(str('kfold/' + img_path[i]), 'test/liquid/' + img_path[i])
    for i in val:
        if label[i] == 2:
            shutil.copyfile(str('kfold/' + img_path[i]), 'val/solid/' + img_path[i])
        elif label[i] == 1:
            shutil.copyfile(str('kfold/' + img_path[i]), 'val/half/' + img_path[i])
        else:
            shutil.copyfile(str('kfold/' + img_path[i]), 'val/liquid/' + img_path[i])
    print(len(os.listdir('train/solid/')), len(os.listdir('train/liquid/')),len(os.listdir('train/half/')))
    print(len(os.listdir('test/solid/')), len(os.listdir('test/liquid/')),len(os.listdir('test/half/')))
    print(len(os.listdir('val/solid/')), len(os.listdir('val/liquid/')),len(os.listdir('val/half/')))


# def split_train(index_train, index_test):#split train to train and val

# label_split=[], img_path_split = [], label_test = []
# for i in index_train:
#     label_split.append(label[i])
#     img_path_split.append(img_path[i])
# for i in index_test:
#     label_test.append(label[i])
# index_train, index_val, label_train, label_val = train_test_split(img_path_split, label_split, test_size=0.2,random_state=10, stratify=label_split)
# return index_train, index_test, index_val


# Initialize model--------------------------------------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
for model_name in model:
    data_dir = os.getcwd()
    # from[resnet, alexnet, inception,vgg]pick one pre-trained model
    num_classes = 3
    batch_size = 8
    num_epochs = 200
    patience = 10
    delta = 0.01
    trial = 0
    feature_extract =False  # false== fine tuning
    img_path = []
    label = []  # 1 is solid, 0 is liquid
    folder_path = ['test', 'train', 'val']
    # Paths for image directory and model
    EVAL_DIR = 'test'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    LOSS = []  # each trial
    ACC = []


    skf = StratifiedKFold(n_splits=5, random_state=10, shuffle=True)
    image_pre()
    create_log_folder()
    write_log('\n' + "patience" + str(patience) + 'delta' + str(delta))

    for train_index, test_index in skf.split(X=img_path, y=label):
        # for train_index, test_index in skf.split(X=img_path,y=label):
        #     train_index, test_index, val_index = split_train(train_index,test_index)

        label_split, img_path_split, label_test = [], [], []
        for i in train_index:
            label_split.append(label[i])
            img_path_split.append(img_path[i])
        for i in test_index:
            label_test.append(label[i])
        train_index, val_index, label_train, label_val = train_test_split(train_index, label_split, test_size=0.2,
                                                                          random_state=10, stratify=label_split)
        print(train_index)
        trial = trial + 1

        write_log('\n' + 'TRIAL' + str(trial))
        # write_log("TRAIN:"+ str(train_index)+ "TEST:"+str(test_index))
        tranfer_file(train_index, test_index, val_index, label)
        model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
        # print(model_ft)
        data_load()

        # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print('GPU is processing'if torch.cuda.is_available() else 'CPU is processing')

        #### Create the Optimizer
        # Send the model to GPU
        model_ft = model_ft.to(device)

        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        params_to_update = model_ft.parameters()
        print("Params to learn:")

        if feature_extract:
            params_to_update = []
            for name, param in model_ft.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t", name)

        else:
            for name, param in model_ft.named_parameters():
                if param.requires_grad == True:
                    print("\t", name)

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

        ###Run Training and Validation Step
        # Setup the loss fxn
        criterion = nn.CrossEntropyLoss()

        # Train and evaluate
        model_ft, hist, h = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs,
                                        patience=patience, delta=delta, is_inception=(model_name == "inception"))
        Path_name = model_name + ('FE' if feature_extract else 'FT')
        path_name = model_name + str(trial) + 'trial' + ('FE' if feature_extract else 'FT')
        PATH = '{}.pth'.format(path_name)
        PATH = os.path.join(Path_name, PATH)
        torch.save(model_ft, PATH)
        # scratch_train()
        plot_history(h)
        print('Trial' + str(trial) + ' training finished')
        write_log('Trial' + str(trial) + ' training finished')
        # empty_imagefolder()#------------------------------------

        # evulate------------------------------------------------------------------------------------------------------------

        EVAL_MODEL = model_ft
        # Load the model for evaluation
        model = torch.load(PATH)
        model.eval()
        # Configure batch size and nuber of cpu's
        num_cpu = multiprocessing.cpu_count()
        bs = 4

        # Prepare the eval data loader
        eval_transform = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])

        eval_dataset = datasets.ImageFolder(root=EVAL_DIR, transform=eval_transform)
        eval_loader = data.DataLoader(eval_dataset, batch_size=bs, shuffle=True,
                                      pin_memory=True, num_workers=0)  # remove num_works

        # Enable gpu mode, if cuda available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Number of classes and dataset-size
        num_classes = len(eval_dataset.classes)
        dsize = len(eval_dataset)

        # Class label names
        class_names = ['liquid', 'half','solid']#0,1,2

        # Initialize the prediction and label lists
        predlist = torch.zeros(0, dtype=torch.long, device='cpu')
        lbllist = torch.zeros(0, dtype=torch.long, device='cpu')
        Path_name = model_name + ('FE' if feature_extract else 'FT')
        Path_name = os.path.join(Path_name, 'trial' + str(trial) + 'wrong predic')
        os.makedirs(Path_name)

        # Evaluate the model accuracy on the dataset
        correct = 0
        total = 0
        count = 0
        test_loss = 0
        with torch.no_grad():
            for images, labels in eval_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                pred = outputs.argmax(dim=1, keepdim=True)  # get
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                test_loss += criterion(outputs, labels).sum().item()  # sum up batch loss
                predlist = torch.cat([predlist, predicted.view(-1).cpu()])
                lbllist = torch.cat([lbllist, labels.view(-1).cpu()])

                import torchvision.transforms.functional as TF

                # Store wrongly predicted images
                wrong_idx = (pred != labels.view_as(pred)).nonzero()[:, 0]
                wrong_samples = images[wrong_idx]
                wrong_preds = pred[wrong_idx]
                actual_preds = labels.view_as(pred)[wrong_idx]

                count += 1
                count_wr_img = 0

                for i in range(len(wrong_idx)):
                    sample = wrong_samples[i]
                    wrong_pred = wrong_preds[i]
                    actual_pred = actual_preds[i]
                    # Undo normalization
                    sample = sample * 0.224
                    sample = sample + 0.456
                    sample = sample * 255.
                    sample = sample.byte()
                    img = TF.to_pil_image(sample)
                    count_wr_img += 1
                    if count_wr_img < 20:
                        PATH = 'No{}_pred{}_actual{}.png'.format(count, wrong_pred.item(), actual_pred.item())
                        PATH = os.path.join(Path_name, PATH)
                        img.save(PATH)

        # print(predlist, lbllist)
        # Overall accuracy
        overall_accuracy = 100 * correct / total
        overall_loss = test_loss / total
        write_log('Accuracy of the network on the {:d} test images: {:.2f}%'.format(dsize,
                                                                                    overall_accuracy))
        write_log('Loss: {:.2f}%'.format(overall_loss))
        print('Accuracy of the network on the {:d} test images: {:.2f}%'.format(dsize,
                                                                                overall_accuracy), 'Loss:',
              str(overall_loss))
        ACC.append(overall_accuracy)
        LOSS.append(overall_loss)
        # Confusion matrix
        conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
        print('Confusion Matrix')
        print('-' * 16)
        print(conf_mat, '\n')
        write_log('Confusion Matrix' + str(conf_mat))

        # Per-class accuracy
        class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)
        print('Per class accuracy')
        print('-' * 18)
        for Label, accuracy in zip(eval_dataset.classes, class_accuracy):
            class_name = Label
            print('Accuracy of class %8s : %0.2f %%' % (class_name, accuracy))
            write_log('Accuracy of class %8s : %0.2f %%' % (class_name, accuracy))
        image_pre()

    import statistics

    write_log('\n' + 'Average and std accuracy,loss')
    print('\n' + 'Average and std accuracy,loss')
    for i in [ACC, LOSS]:
        stdev = statistics.stdev(i)
        mean = sum(i) / len(i)
        write_log('mean: {:.2f}%, std:{:.4f}'.format(mean, stdev))
        print('mean: {:.3f}, std:{:.4f}'.format(mean, stdev))
