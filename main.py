import torch.utils.data as data
from PIL import Image
import torch
import math
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.tensor as tensor
import torchvision.models as models
import os
import torchvision.transforms as transforms
from torch.optim import Adam
class Datasetloader(data.Dataset):
    def __init__(self, root_path, txt_name, transform = None):
        txt_path = root_path + txt_name
        fh = open(txt_path, 'r')
        imgs = []
        label_list = [tensor([1,-1,-1,-1,-1,-1,-1]), tensor([-1,1,-1,-1,-1,-1,-1]),
                    tensor([-1,-1,1,-1,-1,-1,-1]),tensor([-1,-1,-1,1,-1,-1,-1]),
                    tensor([-1,-1,-1,-1,1,-1,-1]),tensor([-1,-1,-1,-1,-1,1,-1]),
                    tensor([-1,-1,-1,-1,-1,-1,1])]
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imags_path = root_path + words[0]
            imgs.append((imags_path, label_list[int(words[1])]))
            self.imgs = imgs
            self.transform = transform
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        # print(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.imgs)

def preprocess(dataset):
    print('==> Preparing data..')
    if str(dataset)=='CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        root_path = './dataset/CIFAR10/'
    elif dataset=='MNIST':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),(0.3081,))
        ])
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),(0.3081,))
        ])
        root_path = './dataset/MNIST/'
    elif dataset=='SVHN':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4391, 0.4457, 0.4740),(0.1990, 0.2020, 0.2034))
        ])
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4391, 0.4457, 0.4740),(0.1990, 0.2020, 0.2034))
        ])
        root_path = './dataset/SVHN/'
    return transform_train,transform_test,root_path




class PUModel(nn.Module):
    def __init__(self, num_classes=7):
        super(PUModel, self).__init__()
        self.features = nn.Sequential(
            # 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),                                 #BN
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),                                #BN
            nn.ReLU(True),
            # 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 6
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 7
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 9
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 10
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 11
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 12
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 13
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=1, stride=1),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

from torch import nn
import torch


class PULoss(nn.Module):
    def __init__(self,  prior, loss=(lambda x: torch.sigmoid(x)), gamma=1, beta=0,
                 ):
        super(PULoss, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.loss_func = loss  # lambda x: (torch.tensor(1., device=x.device) - torch.sign(x))/torch.tensor(2, device=x.device)
        self.unlabeled = torch.tensor([-1,-1,-1,-1,-1,-1.,1.],device='cuda')
        self.min_count = torch.tensor(1.,device='cuda')

    def forward(self, output, target, test=False, theta=0.9):
        assert (output.shape == target.shape)
        #saparate p and u
        unlabeled_ts = self.unlabeled
        unlabeled_ts = unlabeled_ts.repeat(int(len(target.view(-1))/len(unlabeled_ts)))
        positive, unlabeled = [], []
        vec_length = int(len(target.view(-1)) / len(target))
        for i in range(len(target)):
            if bool(target[i][-1] > 0):
                for j in range(vec_length):
                    positive.append([0])
                    unlabeled.append([1])
            else:
                for j in range(vec_length):
                    positive.append([1])
                    unlabeled.append([0])
        positive, unlabeled = torch.tensor(positive, dtype=torch.float, device='cuda').view(-1), \
                              torch.tensor(unlabeled,dtype=torch.float, device='cuda').view(-1)
        n_positive, n_unlabeled = torch.max(self.min_count, torch.sum(positive)), torch.max(self.min_count,
                                                                                            torch.sum(unlabeled))
        #Loss
        target = target.view(-1)
        logit = lambda x: torch.softmax(x, dim=-1)
        logit_value = logit(output).view(-1)
        logit_value = 2*logit_value-1
        loss = lambda x: torch.sigmoid(-x)

        positive_z = logit_value * positive * target
        positive_negative_z = logit_value * positive * unlabeled_ts
        unlabeled_z = logit_value * unlabeled * target

        positive_risk = loss(positive_z) * positive
        positive_negative_risk = loss(positive_negative_z)* positive
        unlabeled_risk = loss(unlabeled_z)* unlabeled

        positive_risk = positive_risk.sum() / n_positive
        positive_negative_risk = positive_negative_risk.sum() / n_positive
        unlabeled_risk = unlabeled_risk.sum()  / n_unlabeled

        prior = theta

        judge_risk = unlabeled_risk - prior * positive_negative_risk#
        pu_risk = prior * positive_risk + max(tensor([0.], device='cuda'), judge_risk)
        if judge_risk > -self.beta:
            return pu_risk
        else:
            return -1*judge_risk

def train(model, device, train_loader, optimizer, epoch,log_interval):
    model.train()
    tr_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss_fct = PULoss(prior=0.6)
        loss = loss_fct(output, target.type(torch.float).view_as(output))
        tr_loss += loss.mean().item()
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    print("Train loss: ", tr_loss)

def test( model, device, test_loader):
    """Testing"""
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss_func = PULoss(prior=0.6)
            test_loss += test_loss_func(output, target.type(torch.float).view_as(output)).item()# sum up batch loss
            logit = lambda x: torch.softmax(x, dim=-1)
            logit_value = logit(output)
            tar_idx = target.argmax(-1)
            tpr_li, fpr_li=[],[]
            threshold_li=[0,0.001,0.005,0.01]
            for i in range(5):
                threshold_li.append(0.2*i+0.1)
            threshold_li.append(0.99)
            threshold_li.append(0.995)
            threshold_li.append(0.999)
            threshold_li.append(1)
            for j in range(len(threshold_li)):
                threshold = tensor(threshold_li[j], device='cuda:0')
                TP, FP, FN, TN, ic_pred,ic_acc = 0, 0, 0, 0, 0,0
                for i in range(len(logit_value)):
                    if bool(logit_value[i][-1] > threshold):
                        if bool(tar_idx[i] == tensor(6, device='cuda:0')):
                            TP += 1
                        else:
                            FP += 1
                    else:
                        if bool(tar_idx[i] != tensor(6, device='cuda:0')):
                            TN += 1
                        else:
                            FN += 1
                tpr_li.append(TP/max(TP+FN,1))
                fpr_li.append(FP/max(TN+FP,1))
    auc=0
    for i in range(len(tpr_li)-1):
        auc+=0.5*(fpr_li[i+1]-fpr_li[i])*(tpr_li[i]+tpr_li[i+1])
    print('AUC for current epoch:',round(-1*auc,5))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default='CIFAR10',
                        type=str,
                        required=False,
                        help="Select dataset")
    parser.add_argument("--lr",
                        default=1e-5,
                        type=float,
                        required=False,
                        help="Select learning rate")
    parser.add_argument("--weight_decay",
                        default=0.005,
                        type=float,
                        required=False,
                        help="Select weight decay parameter")
    parser.add_argument("--output_dir",
                        default='./output',
                        type=str,
                        required=False,
                        help="Output directory")
    parser.add_argument("--device",
                        default='cuda',
                        type=str,
                        required=False,
                        help="Default is cuda")
    parser.add_argument("--num_workers",
                        default=0,
                        type=int,
                        required=False,
                        help="Dataloader parameter, default is 0")
    parser.add_argument("--pin_memory",
                        default=True,
                        type=bool,
                        required=False,
                        help="True stands for open pin memory")
    parser.add_argument("--num_epoch",
                        default=400,
                        type=int,
                        required=False,
                        help="Number of epochs, default is 400")
    parser.add_argument("--batch_size",
                        default=500,
                        type=int,
                        required=False,
                        help="Number of batch size, default is 500")
    args = parser.parse_args()
    dataset=args.dataset
    if dataset not in ['MNIST','CIFAR10','SVHN']:
        raise ValueError('Dataset must be selected within MNIST, CIFAR10 or SVHN.')
    transform_train,transform_test,root_path=preprocess(dataset)
    output_dir=args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, str(args.dataset)+'.bin')
    if args.device=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is not available.')
    device = torch.device(args.device)
    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if torch.cuda.is_available() else {}
    train_set = Datasetloader(root_path = root_path,txt_name = "train.txt",transform = transform_train)
    train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size=args.batch_size, shuffle=True,**kwargs)
    test_set = Datasetloader(root_path = root_path,txt_name = "val.txt",transform = transform_test)
    test_loader = torch.utils.data.DataLoader(dataset = test_set, batch_size=1000, shuffle=True,**kwargs)
    model = PUModel().to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_train_epochs=args.num_epoch
    if args.dataset=='CIFAR10':
        num_tr_sample=39000
    elif args.dataset=='MNIST':
        num_tr_sample =46000
    elif args.dataset=='SVHN':
        num_tr_sample =57000
    log_interval=math.floor(num_tr_sample/2/args.batch_size)
    for epoch in range(1, num_train_epochs + 1):
        train(model, device, train_loader, optimizer, epoch, log_interval)
        test(model, device, test_loader)
        torch.save(model.state_dict(), output_model_file)


if __name__ == "__main__":
    main()




