# Autograd Mechanics

## What is Autograd?

Autograd is the component of PyTorch responsible for automatic differentiation. In simple terms, it keeps track of every operation performed on a tensor so that it can later compute the gradient of some output with respect to that tensor. This is the mechanism that makes training neural networks through gradient descent practical, since it removes the need for developers to manually derive and implement gradient formulas for every operation in a model.

When a tensor is created with gradient tracking enabled, PyTorch builds a computation graph behind the scenes as operations are applied to it. Each node in this graph represents an operation, and the graph records enough information to compute gradients through the chain rule once the final output is known.

## Enabling Gradient Tracking

A tensor only participates in autograd if it is explicitly marked to require gradients. This is typically done when creating the tensor:

```python
import torch

x = torch.randn(3, requires_grad=True)
y = x * 2
z = y.sum()
```

In this example, `x` is a tensor that requires gradients. Any tensor derived from operations on `x`, such as `y` and `z`, will also track gradient information automatically, since PyTorch propagates this requirement forward through the computation graph.

## Computing Gradients

Once a scalar output has been computed, calling the backward method triggers the gradient computation. PyTorch walks the computation graph in reverse, applying the chain rule at each step to compute how much each input contributed to the final output.

```python
z.backward()
print(x.grad)
```

After this call, the `grad` attribute of `x` contains the gradient of `z` with respect to `x`. This gradient can then be used by an optimizer to update the values of `x`, which in a real model would typically be the learnable weights.

## The Computation Graph is Dynamic

One of the distinguishing features of PyTorch's autograd system, compared to some other frameworks, is that the computation graph is built dynamically at runtime rather than being defined ahead of time. This means the structure of the graph can change from one execution to the next, which makes it straightforward to use standard Python control flow, such as loops and conditionals, directly inside model code.

## Disabling Gradient Tracking

There are situations where gradient tracking is unnecessary and only adds computational overhead, such as during model evaluation or inference. In these cases, gradient tracking can be temporarily disabled:

```python
with torch.no_grad():
    predictions = model(input_data)
```

Disabling gradient tracking in this way reduces memory usage and speeds up computation, since PyTorch no longer needs to maintain the history required to compute gradients.
