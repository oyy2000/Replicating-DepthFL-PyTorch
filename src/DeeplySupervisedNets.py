import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Define a simple CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 128 * 8, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Define the Deeply Supervised Network
class DeeplySupervisedNetwork(nn.Module):
    def __init__(self, num_classes, num_hidden_layers):
        super(DeeplySupervisedNetwork, self).__init__()
        self.base_network = SimpleCNN(num_classes)
        self.supervision_layers = nn.ModuleList([nn.Linear(512, num_classes) for _ in range(num_hidden_layers)])
        
    def forward(self, x):
        hidden_outputs = []
        for layer in self.base_network.features:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                hidden_outputs.append(x)
        
        logits = self.base_network.classifier(x)
        supervision_logits = [layer(x.view(x.size(0), -1)) for layer in self.supervision_layers]
        return logits, supervision_logits


# Set up data loading and preprocessing (you can adapt this for your dataset)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

# Initialize the DSN
num_classes = 10
num_hidden_layers = 2
dsn = DeeplySupervisedNetwork(num_classes, num_hidden_layers)

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(dsn.parameters(), lr=0.001, momentum=0.9)

# Training loop
for epoch in range(5):  # Change the number of epochs as needed
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs, supervision_outputs = dsn(inputs)
        loss = criterion(outputs, labels)

        for sup_out in supervision_outputs:
            loss += criterion(sup_out, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0

print("Finished Training")
