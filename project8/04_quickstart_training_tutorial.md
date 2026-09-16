# Quickstart: Training a Simple Model

This tutorial walks through the basic steps required to train a neural network in PyTorch, from defining a model to running a full training loop.

## Step 1: Define the Model

Models in PyTorch are typically defined as classes that inherit from `nn.Module`. The class needs to define its layers in the constructor and describe how data flows through those layers in a method called `forward`.

```python
import torch
import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x
```

## Step 2: Set Up the Loss Function and Optimizer

The loss function measures how far the model's predictions are from the true labels, while the optimizer defines how the model's parameters are updated based on the computed gradients.

```python
model = SimpleClassifier(input_size=10, hidden_size=32, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

## Step 3: Write the Training Loop

A typical training loop repeats the same sequence of steps for a fixed number of epochs: run the input data through the model, compute the loss, compute gradients, and update the model's parameters.

```python
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(train_inputs)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

It is important to call `optimizer.zero_grad()` at the start of each iteration. Gradients in PyTorch accumulate by default, meaning that without this reset, gradients from previous iterations would be added to the current ones, leading to incorrect updates.

## Step 4: Evaluate the Model

After training, the model can be evaluated on unseen data. During evaluation, gradient tracking should be disabled, and the model should be switched to evaluation mode, which changes the behavior of certain layers such as dropout.

```python
model.eval()
with torch.no_grad():
    test_outputs = model(test_inputs)
    predictions = torch.argmax(test_outputs, dim=1)
    accuracy = (predictions == test_labels).float().mean()
    print(f"Test Accuracy: {accuracy.item():.4f}")
```

## Common Pitfalls

A few mistakes come up frequently for those training their first PyTorch model. Forgetting to zero the gradients before each backward pass is one of the most common, since it silently corrupts training without raising an error. Another common issue is forgetting to switch between `model.train()` and `model.eval()` modes, which can lead to inconsistent behavior if the model contains layers that behave differently during training versus inference, such as dropout or batch normalization layers.
