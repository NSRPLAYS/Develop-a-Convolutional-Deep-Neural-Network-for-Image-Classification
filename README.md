# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Include the Problem Statement and Dataset.

## Neural Network Model
<img width="1039" height="754" alt="image" src="https://github.com/user-attachments/assets/60b76dac-5d2c-49c3-a18f-35285600a6ce" />

## DESIGN STEPS

### STEP 1:
Import the required libraries (torch, torchvision, torch.nn, torch.optim) and load the image dataset with necessary preprocessing like normalization and transformation.

### STEP 2:
Split the dataset into training and testing sets and create DataLoader objects to feed images in batches to the CNN model.

### STEP 3:
Define the CNN architecture using convolutional layers, ReLU activation, max pooling layers, and fully connected layers as implemented in the CNNClassifier class.

### STEP 4:
Initialize the model, define the loss function (CrossEntropyLoss), and choose the optimizer (Adam) for training the network.

### STEP 5:
Train the model using the training dataset by performing forward pass, computing loss, backpropagation, and updating weights for multiple epochs.

### STEP 6:
Evaluate the trained model on test images and verify the classification accuracy for new unseen images.


## PROGRAM

### Name:Naveen Sairam B

### Register Number: 21224240103

```python
class CNNClassifier(nn.Module):
    def __init__(self, input_size):
        super(CNNClassifier, self).__init__()
       self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)
    def forward(self, x):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(torch.relu(self.conv3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x


# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

## Step 3: Train the Model
def train_model(model, train_loader, num_epochs=3):
  for epoch in range(num_epochs):
    model.train()
    running_loss=0.0
    for images, labels in train_loader:
      optimizer.zero_grad()
      outputs=model(images)
      loss = criterion(outputs,labels)
      loss.backward()
      optimizer.step()
      running_loss +=loss.item()
#print('Name: Naveen Sairam B')
#print('Register Number: 212224240103')
      print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
```

### OUTPUT

## Training Loss per Epoch

Epoch [1/3], Loss: 0.2115
Epoch [2/3], Loss: 0.1855
Epoch [3/3], Loss: 0.1683

## Confusion Matrix

<img width="948" height="828" alt="image" src="https://github.com/user-attachments/assets/d09f3474-82d0-418c-a993-4fdbec92cdaa" />


## Classification Report
<img width="580" height="437" alt="image" src="https://github.com/user-attachments/assets/197c75ba-6aa3-4ab2-bbb7-0a149299ccc4" />


### New Sample Data Prediction
<img width="561" height="621" alt="image" src="https://github.com/user-attachments/assets/1636b158-3d35-4088-9c68-60258fb5f59e" />


## RESULT
To Develop a Convolutional Deep Neural Network for Image Classification has been done and the output has been verified.
