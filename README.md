# EasyDeL 

EasyDeL (Easy Deep Learning) is an open-source library designed to accelerate and optimize the training process of machine learning models. This library is primarily focused on PyTorch, and the team behind EasyDeL is planning to add support for JAX/Flax and TensorFlow in the future.

## Installation
To install EasyDeL, you can use pip:

```bash
pip install easydel
```

## Usage
To use EasyDeL in your project, you will need to import the library in your Python script and use its various functions and classes. Here is an example of how to import EasyDeL and use its Optimizer class:

```python

import EasyDeL

optimizer = EasyDeL.Optimizer(model.parameters(), lr=0.001, {...})
model = EasyDeL.Model(model)

```
The Optimizer class in EasyDeL is similar to the torch.optim.Optimizer class in PyTorch, but with added features to optimize your training process further.

## Contributing
EasyDeL is an open-source project, and contributions are welcome. If you would like to contribute to EasyDeL, please fork the repository, make your changes, and submit a pull request. The team behind EasyDeL will review your changes and merge them if they are suitable.

## License
EasyDeL is released under the Apache v2 license. Please see the LICENSE file in the root directory of this project for more information.

## Contact
If you have any questions or comments about EasyDeL, you can reach out to the team behind EasyDeL
