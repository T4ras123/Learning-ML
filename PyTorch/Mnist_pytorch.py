import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms



# Parameters for the model

input_size = 784  # 28x28 image as a 1D vector
hidden_size = 500 # Hidden layer size
num_classes = 10  # 10 classes, one for each digit
num_epochs = 10 
batch_size = 100 
learning_rate = 0.001 

# Load the MNIST dataset

train = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())


# Data loader

train_l = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
test_l = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

# Neural network, fully connected with one hidden layer 

class mnistNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(mnistNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
    
model = mnistNet(input_size, hidden_size, num_classes)

# Loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_l)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_l):
        
        images = images.reshape(-1, 28*28)
        labels = labels
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # backward prop and optimize
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))



with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_l:
        images = images.reshape(-1, 28*28)
        labels = labels
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')