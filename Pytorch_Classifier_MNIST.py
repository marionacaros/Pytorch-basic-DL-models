
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

transform = transforms.ToTensor()

mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

print(mnist_trainset)

batch_size = 20
epochs = 5


# #### Make the dataset iterable
trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=batch_size,shuffle=False)

classes = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9']

print('There are {} images in the train set'.format(len(mnist_trainset)))
print('There are {} images in the test set'.format(len(mnist_testset)))
print('There are {} batches in the train set'.format(len(trainloader)))
print('There are {} batches in the test set'.format(len(testloader)))


# #### Show image

batch = next(iter(testloader))
samples = batch[0][:5]
y_true = batch[1]
for i, sample in enumerate(samples):
    plt.subplot(1,5,i+1)
    plt.title('Number: %i' % y_true[i])
    plt.imshow(sample.numpy().reshape((28,28)))
    plt.axis('off')


# ### Create the model

import torch.nn as nn

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # padding = [(filter size - 1)/2] -> input size = output size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        
        # Output size of each of 8 feature maps is
        # [(input_size - filter_size + 2*(padding)/ stride) + 1]  (28-3+2)/1 + 1 = 28
        # Batch normlaization
        self.batchnorm1 = nn.BatchNorm2d(8) # out_channels = 8
        # ReLu
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) 
        # output of each feature map is now 28/2 = 14
        
        # padding= (5 - 1) / 2 = 2 -> input size = output size
        self.conv2 = nn.Conv2d(8, 32, 5,  stride=1, padding=2)
        # Output size of each of the 32 feature maps remains 14
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) 
        # After max pooling, output of each feature map = 14/2 = 7

        # Flatten the feature maps
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(32 * 7 * 7, 600)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(600, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        # Flatten the output it will take the shape (batch_size, num_features)
        num_feat = 32 * 7 * 7
        x = x.view(-1, num_feat) # reshape
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = CNN()
print(net)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


# ### Train the network

for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

