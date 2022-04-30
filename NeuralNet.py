import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 64)
        #self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.flatten(start_dim=1)
        #print(x.shape)
        #x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)





def testNet(test_set):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_set:
            X, labels = data
            output = net(X.view(-1, 3, 32, 32))
            for idx, i in enumerate(output):
                if torch.argmax(i) == labels[idx]:
                    correct += 1
                total += 1
    #print(f"Accuracy of the network on the 10000 test images: {100 * correct / total}%")
    print(f"Accuracy of the network on the provided test images: {100 * correct / total}%")



def main(net, trainloader, testloader):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    EPOCH = 20

    for epoch in range(EPOCH):
        for data in trainloader:
            X, labels = data
            net.zero_grad()
            output = net(X.view(-1, 3, 32, 32))
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()
        print(epoch+1, ": loss="+str(round(loss.item(), 3)))

    testNet(testloader)
    testNet(trainloader)

    









if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root='./Desktop/UNI/SecondYear/INT2/assessment/data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./Desktop/UNI/SecondYear/INT2/assessment/data', train=False,download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=2)

    net = Net()
    #net.forward(torch.randn(1, 3, 50, 50))

    main(net, trainloader, testloader)
