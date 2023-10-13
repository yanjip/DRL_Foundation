import torch
import matplotlib.pyplot as plt
from torchvision import datasets,transforms
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F

# prepare dataset

batch_size=64

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

train_dataset=datasets.MNIST(root='../dataset/mnist',
                             train=True,
                             download=True,
                             transform=transform)

train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

test_dataset=datasets.MNIST(root='../dataset/mnist',
                            train=False,
                            download=True,
                            transform=transform)

test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

# design model

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=torch.nn.Conv2d(1,10,kernel_size=5)
        self.conv2=torch.nn.Conv2d(10,20,kernel_size=5)
        self.pooling=torch.nn.MaxPool2d(kernel_size=2)
        self.linear1=torch.nn.Linear(320,16)
        self.linear2=torch.nn.Linear(16,10)

        pass
    def forward(self,x):
        x=F.relu(self.pooling(self.conv1(x)))
        x=F.relu(self.pooling(self.conv2(x)))
        x=x.view(batch_size,-1)
        x=F.relu(self.linear1(x))
        x=self.linear2(x)

        return x
    pass

model=Net()
    # loss and optimizer
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

    # training cycling

def train(epoch):
    run_lossing=0
    for id,data in enumerate(train_loader,0):
        inputs,labels=data
        outputs=model(inputs)
        loss=criterion(outputs,labels)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        run_lossing+=loss
        if id%300==299:
            print('%d %5d %4f'%(epoch,id,run_lossing))
            run_lossing=0
acurracy=[]
def test():
    correct=0
    total=0
    with torch.no_grad():
        for data in test_loader:
            inputs,labels=data
            outputs=model(inputs)
            _,predicted=torch.max(outputs,dim=1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    acurracy.append(100*correct/total)

if __name__=='__main__':
    for epoch in range(10):
        train(epoch)
        test()
    plt.plot(range(10),acurracy)
    plt.xlabel('epoch')
    plt.ylabel('acurracy')
    plt.show()
