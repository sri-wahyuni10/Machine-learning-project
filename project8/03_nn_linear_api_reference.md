# torch.nn.Linear

## Description

Applies a linear transformation to incoming data. This layer computes an output by multiplying the input by a weight matrix and adding a bias term. It is one of the most commonly used building blocks in neural network architectures, often referred to as a fully connected layer or dense layer.

The transformation performed can be described as: output equals input multiplied by the transpose of the weight matrix, plus the bias vector.

## Parameters

- **in_features** (int): The size of each input sample. This must match the size of the last dimension of the input tensor.
- **out_features** (int): The size of each output sample. This determines the size of the last dimension of the output tensor.
- **bias** (bool, optional): If set to `False`, the layer will not learn an additive bias term. Default is `True`.
- **device** (optional): The device on which to allocate the layer's parameters, such as CPU or a specific GPU.
- **dtype** (optional): The data type to use for the layer's parameters, such as 32-bit or 64-bit floating point.

## Shape

- **Input**: A tensor of shape `(*, in_features)`, where `*` represents any number of leading dimensions, such as a batch dimension.
- **Output**: A tensor of shape `(*, out_features)`, preserving the leading dimensions from the input.

## Attributes

- **weight**: The learnable weights of the layer, with shape `(out_features, in_features)`. These values are initialized from a uniform distribution and updated during training.
- **bias**: The learnable bias of the layer, with shape `(out_features,)`. Only present if `bias` is set to `True`.

## Example

```python
import torch
import torch.nn as nn

layer = nn.Linear(in_features=20, out_features=10)
input_tensor = torch.randn(32, 20)
output_tensor = layer(input_tensor)
print(output_tensor.shape)
```

Running this example produces an output tensor of shape `(32, 10)`, since the batch dimension of 32 is preserved while the feature dimension changes from 20 to 10 according to the layer's configuration.

## Notes

Linear layers are frequently stacked together, often separated by non-linear activation functions, to build multi-layer neural networks. Without an activation function between them, multiple linear layers stacked in sequence would be mathematically equivalent to a single linear layer, since the composition of linear transformations is itself linear.
