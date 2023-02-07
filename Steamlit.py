import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import PIL
import numpy as np


# Load the pre-trained ResNet18 model
model_1 = torchvision.models.resnet18(pretrained=True)

# Freeze all layers in the feature extractor
for param in model_1.parameters():
    param.requires_grad = False

# Replace the last layer with a fully connected layer for 10 classes
model_1.fc = torch.nn.Sequential(
    torch.nn.Linear(512, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 10)
)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Set the optimizer to SGD with a learning rate of 0.01
optimizer = optim.SGD(model_1.parameters(), lr=0.01)


# Load the CIFAR-10 dataset and perform data augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the training set
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Create a DataLoader for the training set
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

# Load the test set
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create a DataLoader for the test set
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)




acc1 = []
epochs1 = 0

# Train the model
for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model_1(inputs)        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    print('Epoch [{}/10], Loss: {:.4f}'.format(epoch + 1, running_loss / len(trainloader)))
    epochs1 = epoch + 1



print('Finished Training')


# Evaluate the model on the test set
  
with torch.no_grad():
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    for data, target in testloader:
        output = model_1(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        all_predictions.extend(predicted.tolist())
        all_targets.extend(target.tolist())

    accuracy = 100 * correct / total
    print('Accuracy of the network on the test set: {:.2f}%'.format(accuracy))
    acc1.append(accuracy)
    
    # Calculate the confusion matrix
    cm1 = confusion_matrix(all_targets, all_predictions)
    #print('Confusion matrix:')
    #print(cm)
    f1_1 = f1_score(all_targets, all_predictions, average='weighted')
    print('F1 score: {:.2f}'.format(f1_1))


sns.heatmap(cm1, annot=True, fmt="d")
plt.xlabel("Prediction")
plt.ylabel("True Label")
plt.show()

"""# Different Hyperparameters : Model 2
Learning rate: 0.1, Batch size: 128, Optimizer: Adam, Epochs: 20, Activation function: Tanh
"""

# Load the pre-trained ResNet18 model
model_2 = torchvision.models.resnet18(pretrained=True)

# Freeze all layers in the feature extractor
for param in model_2.parameters():
    param.requires_grad = False

# Replace the last layer with a fully connected layer for 20 classes
model_2.fc = torch.nn.Sequential(
    torch.nn.Linear(512, 512),
    torch.nn.Tanh(),
    torch.nn.Linear(512, 10)
)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Set the optimizer to SGD with a learning rate of 0.1
optimizer = optim.Adam(model_2.parameters(), lr=0.1)


# Load the CIFAR-10 dataset and perform data augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the training set
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Create a DataLoader for the training set
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

# Load the test set
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create a DataLoader for the test set
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)


acc2 = []
epochs2 = 0


# Train the model
for epoch in range(20):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model_2(inputs)        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    print('Epoch [{}/20], Loss: {:.4f}'.format(epoch + 1, running_loss / len(trainloader)))
   
    epochs2 = epoch + 1



print('Finished Training')

# Evaluate the model on the test set
with torch.no_grad():
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    for data, target in testloader:
        output = model_2(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        all_predictions.extend(predicted.tolist())
        all_targets.extend(target.tolist())
    accuracy = 100 * correct / total
    print('Accuracy of the network on the test set: {:.2f}%'.format(accuracy))
    acc2.append(accuracy)
    # Calculate the confusion matrix
    cm2 = confusion_matrix(all_targets, all_predictions)
    f1_2 = f1_score(all_targets, all_predictions, average='weighted')
    print('F1 score: {:.2f}'.format(f1_2))

sns.heatmap(cm2, annot=True, fmt="d")
plt.xlabel("Prediction")
plt.ylabel("True Label")
plt.show()

"""#Different Hyperparameters: Model 3
Learning rate: 0.001, Batch size: 256, Optimizer: Adagrad, Epochs: 30, Activation function: Sigmoid
"""

# Load the pre-trained ResNet18 model
model_3 = torchvision.models.resnet18(pretrained=True)

# Freeze all layers in the feature extractor
for param in model_3.parameters():
    param.requires_grad = False

# Replace the last layer with a fully connected layer for 30 classes
model_3.fc = torch.nn.Sequential(
    torch.nn.Linear(512, 512),
    torch.nn.Sigmoid(),
    torch.nn.Linear(512, 10)
)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Set the optimizer to SGD with a learning rate of 0.001
optimizer = optim.Adagrad(model_3.parameters(), lr=0.001)


# Load the CIFAR-10 dataset and perform data augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the training set
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Create a DataLoader for the training set
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

# Load the test set
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create a DataLoader for the test set
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)



acc3 = []
epochs3 = 0

# Train the model
for epoch in range(30):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model_3(inputs)        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    print('Epoch [{}/30], Loss: {:.4f}'.format(epoch + 1, running_loss / len(trainloader)))
   
    epochs3 = epoch + 1


print('Finished Training')


# Evaluate the model on the test set
with torch.no_grad():
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    for data, target in testloader:
        output = model_3(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        all_predictions.extend(predicted.tolist())
        all_targets.extend(target.tolist())
    accuracy = 100 * correct / total
    print('Accuracy of the network on the test set: {:.2f}%'.format(accuracy))
    acc3.append(accuracy)
    # Calculate the confusion matrix
    cm3 = confusion_matrix(all_targets, all_predictions)
    f1_3 = f1_score(all_targets, all_predictions, average='weighted')
    print('F1 score: {:.2f}'.format(f1_3))

sns.heatmap(cm3, annot=True, fmt="d")
plt.xlabel("Prediction")
plt.ylabel("True Label")
plt.show()

"""#Different Hyperparameters : Model 4
Learning rate: 0.05, Batch size: 512, Optimizer: RProp, Epochs: 40, Activation function: Leaky ReLU

"""

# Load the pre-trained ResNet18 model
model_4 = torchvision.models.resnet18(pretrained=True)

# Freeze all layers in the feature extractor
for param in model_4.parameters():
    param.requires_grad = False

# Replace the last layer with a fully connected layer for 40 classes
model_4.fc = torch.nn.Sequential(
    torch.nn.Linear(512, 512),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(512, 10)
)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Set the optimizer to SGD with a learning rate of 0.05
optimizer = optim.Rprop(model_4.parameters(), lr=0.05)


# Load the CIFAR-10 dataset and perform data augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the training set
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Create a DataLoader for the training set
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=2)

# Load the test set
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create a DataLoader for the test set
testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=2)


acc4 = []
epochs4 = 0


# Train the model
for epoch in range(40):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model_4(inputs)        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    print('Epoch [{}/40], Loss: {:.4f}'.format(epoch + 1, running_loss / len(trainloader)))
    epochs4 = epoch + 1



print('Finished Training')


# Evaluate the model on the test set
with torch.no_grad():
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    for data, target in testloader:
        output = model_4(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        all_predictions.extend(predicted.tolist())
        all_targets.extend(target.tolist())
    accuracy = 100 * correct / total
    print('Accuracy of the network on the test set: {:.2f}%'.format(accuracy))
    acc4.append(accuracy)
    # Calculate the confusion matrix
    cm4 = confusion_matrix(all_targets, all_predictions)

    f1_4 = f1_score(all_targets, all_predictions, average='weighted')
    print('F1 score: {:.2f}'.format(f1_4))


sns.heatmap(cm4, annot=True, fmt="d")
plt.xlabel("Prediction")
plt.ylabel("True Label")
plt.show()

"""#Different Hyperparameteers: Model 5
Learning rate: 0.005, Batch size: 128, Optimizer: Adamax, Epochs: 50, Activation function: ELU
"""

# Load the pre-trained ResNet18 model
model_5 = torchvision.models.resnet18(pretrained=True)

# Freeze all layers in the feature extractor
for param in model_5.parameters():
    param.requires_grad = False

# Replace the last layer with a fully connected layer for 50 classes
model_5.fc = torch.nn.Sequential(
    torch.nn.Linear(512, 512),
    torch.nn.ELU(),
    torch.nn.Linear(512, 10)
)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Set the optimizer to SGD with a learning rate of 0.005
optimizer = optim.Adamax(model_5.parameters(), lr=0.005)


# Load the CIFAR-10 dataset and perform data augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the training set
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Create a DataLoader for the training set
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

# Load the test set
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create a DataLoader for the test set
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)



acc5 = []
epochs5 = 0

# Train the model
for epoch in range(50):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model_5(inputs)        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    print('Epoch [{}/50], Loss: {:.4f}'.format(epoch + 1, running_loss / len(trainloader)))
    epochs5 = epoch + 1


print('Finished Training')


# Evaluate the model on the test set
with torch.no_grad():
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    for data, target in testloader:
        output = model_5(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        all_predictions.extend(predicted.tolist())
        all_targets.extend(target.tolist())
    accuracy = 100 * correct / total
    print('Accuracy of the network on the test set: {:.2f}%'.format(accuracy))
    acc5.append(accuracy)
    # Calculate the confusion matrix
    cm5 = confusion_matrix(all_targets, all_predictions)
    f1_5 = f1_score(all_targets, all_predictions, average='weighted')
    print('F1 score: {:.2f}'.format(f1_5))

sns.heatmap(cm5, annot=True, fmt="d")
plt.xlabel("Prediction")
plt.ylabel("True Label")
plt.show()






def predict_class(model, image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    
    output = model(image)
    _, prediction = torch.max(output.data, 1)
    
    return prediction.item()


models = [model_1, model_2, model_3, model_4, model_5]
model_names = ["Model 1", "Model 2", "Model 3", "Model 4", "Model 5"]

def main():
    st.title("Image Classifier")
    st.set_page_config(page_title="Image Classifier", page_icon=":camera:", layout="wide")

    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if image_file is not None:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        selected_model = st.selectbox("Choose a model", model_names)
        selected_index = model_names.index(selected_model)
        selected_model = models[selected_index]

        prediction = predict_class(selected_model, image_file)
        st.write("The model predicts the image as:", classes[prediction])

if __name__ == '__main__':
    main()