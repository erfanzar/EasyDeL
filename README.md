#EasyDL 
EasyDL is an open-source library designed to accelerate and optimize the training process of machine learning models. This library is primarily focused on PyTorch, and the team behind EasyDL is planning to add support for JAX/Flax and TensorFlow in the future.

## Installation
To install EasyDL, you can use pip:

```bash
pip install easydl
```

## Usage
To use EasyDL in your project, you will need to import the library in your Python script and use its various functions and classes. Here is an example of how to import EasyDL and use its Optimizer class:

```python

import EasyDL

optimizer = EasyDL.Optimizer(model.parameters(), lr=0.001, {...})
model = EasyDL.Model(model)

```
The Optimizer class in EasyDL is similar to the torch.optim.Optimizer class in PyTorch, but with added features to optimize your training process further.

## Contributing
EasyDL is an open-source project, and contributions are welcome. If you would like to contribute to EasyDL, please fork the repository, make your changes, and submit a pull request. The team behind EasyDL will review your changes and merge them if they are suitable.

## License
EasyDL is released under the Apache v2 license. Please see the LICENSE file in the root directory of this project for more information.

Contact
If you have any questions or comments about EasyDL, you can reach out to the team behind EasyDL
