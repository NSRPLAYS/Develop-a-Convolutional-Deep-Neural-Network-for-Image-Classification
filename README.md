# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Include the Problem Statement and Dataset.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 2:
Build the CNN model using convolution, pooling, and fully connected layers.

### STEP 3:
Initialize the model, define loss function, and select optimizer.

### STEP 4:
Train the CNN model using training data for several epochs.

### STEP 5:
Test the model using accuracy and confusion matrix.

### STEP 6:
Predict and display the class label of a single test image.




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

Include the Training Loss per epoch

## Confusion Matrix

Include confusion matrix here

## Classification Report
Include classification report here

### New Sample Data Prediction
Include your sample input and output here

## RESULT
Include your result here
