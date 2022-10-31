import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from tqdm import tqdm

from torchsummary import summary

trans = transforms.Compose([transforms.ToTensor()])
train_mnist = datasets.MNIST('./', train=True, download=True, transform=trans)
test_mnist = datasets.MNIST('./', train=False, download=True, transform=trans)

train_dl = DataLoader(train_mnist, batch_size=64, shuffle=True, num_workers=8)
test_dl = DataLoader(test_mnist, batch_size=64, shuffle=True, num_workers=8)


def display_ds(train_dl):
    train_bs = next(iter(train_dl))
    for i in range(8):
        plt.subplot(2, 4, i+1)
        plt.title(str(train_bs[1][i]))
        plt.axis('off')
        plt.imshow(train_bs[0][i].permute(1, 2, 0).numpy())
    plt.show()


display_ds(train_dl)


class FC_net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 10)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout()

    def forward(self, inputs):
        x = self.flatten(inputs)
        x = self.relu(self.fc1(x))
        x = self.drop(self.relu(self.fc2(x)))
        x = self.drop(self.relu(self.fc3(x)))
        x = F.softmax(self.fc4(x), dim=1)

        return x


net = FC_net().cuda()
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(net.parameters(), lr=0.0001)

print('='*49)
print(summary(net, (1, 28, 28)))
print('='*49)

train_loss = []
train_acc = []
test_loss = []
test_acc = []

net.train()
for epoch in range(20):
    train_epoch_loss = 0
    train_epoch_acc = 0
    test_epoch_loss = 0
    test_epoch_acc = 0

    train_tqdm = tqdm(train_dl, desc=f'Epoch: {epoch+1}')
    for x, y in train_tqdm:
        x, y = x.cuda(), y.cuda()
        y_pred = net(x)
        loss = criterion(y_pred, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        with torch.no_grad():
            # print(y_pred)
            train_epoch_acc += (torch.argmax(y_pred, dim=1)
                                == y).sum().item() / len(y)
            train_epoch_loss += loss.item()

    train_epoch_loss /= len(train_dl)
    train_epoch_acc /= len(train_dl)

    print(f'Train Loss: {train_epoch_loss:.4f};',
          f'Train Accuracy: {train_epoch_acc:.4f}', sep='\t')

    train_loss.append(train_epoch_loss)
    train_acc.append(train_epoch_acc)

    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.cuda(), y.cuda()
            y_test_pred = net(x)
            test_loss_ = criterion(y_test_pred, y)

            test_epoch_loss += test_loss_.item()
            test_epoch_acc += (torch.argmax(y_test_pred, dim=1)
                               == y).sum().item() / len(y)

        test_epoch_loss /= len(test_dl)
        test_epoch_acc /= len(test_dl)

        print(f'Test Loss: {test_epoch_loss:.4f};',
              f'Test Accuracy: {test_epoch_acc:.4f}', sep='\t')

        test_loss.append(test_epoch_loss)
        test_acc.append(test_epoch_acc)

# print(train_loss, train_loss, test_acc, test_loss)

plt.subplot(2, 2, 1)
plt.title('train_loss')
plt.plot(train_loss)

plt.subplot(2, 2, 2)
plt.title('train_acc')
plt.plot(train_acc)

plt.subplot(2, 2, 3)
plt.title('test_loss')
plt.plot(test_loss)

plt.subplot(2, 2, 4)
plt.title('test_acc')
plt.plot(test_acc)

save = False
if save:
    plt.savefig('./metrics.png')
plt.show()
