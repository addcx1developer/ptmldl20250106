# Deep Learning & Machine Learning
### Anatomy of Neural Networks

- **Units/Neurons** - each layer nodes (e.g. two nodes is equal to two neurons)
- **Input Layer** - data goes in here
- **Hidden Layer** - learns patterns in data
- **Output Layer** - outputs learned representation or prediction probabilities

Each layer is usually combination of **linear (straight line)** and/or **non-linear (not straight line)** functions

> [!NOTE]
> "patterns" is an arbitrary term, you'll often hear "embedding", "weights", "feature representation", "feature vectors" all referring to similar things.

### Types of Learning
- **Supervised Learning**
- **Unsupervised & Self-supervised Learning**
- **Transfer Learning**
- **Reinforcement Learning**

**Natural Language Processing (NLP)** - combines computational linguistics, machine learning, and deep learning models to process human language

### [PyTorch](https://pytorch.org)
- Most popular research deep learning framework
- Write fast deep learning code in Python (able to run on a GPU/many GPUs)
- Able to access many pre-built deep learning models (Torch Hub/torchvision.models)
- Whole stack: preprocess data, model data, deploy model in your application/cloud
- Originally designed and used in-house by Facebook/Meta (now open-source and used by companies such as Tesla, Microsoft, OpenAI)

###
1. Get data ready (turn into tensors)
2. Build or pick a pretrained model (to suit your problem)\
  2.1. Pick a loss function & optimizer\
  2.2. Build a training loop
3. Fit the model to the data and make a prediction
4. Evaluate the model
5. Improve through experimentation
6. Save and reload your trained model

### Three datasets
possibly the most important concept in machine learning...

1. Course materials (training set)\
  Model learns patterns from here
2. Practice exam (validation set)\
  Tune model patterns
3. Final exam (test set)\
  See if the model is ready for the wild

**Generalization**\
The ability for a machine learning model to perform well on data it hasn't seen before.

**Subclass nn.Module**\
this contains all the building blocks for neural networks

**Initialise model parameters**\
to be used in various computations (these could be different layers from torch.nn, single parameters, hard-coded values or functions)

**requires_grad=True**\
PyTorch will track the gradients of this specific parameter for use with torch.autograd and gradient descent (for many torch.nn modules, requires_grad=True is set by default)

> [!NOTE]
> Any subclass of nn.Module needs to override *forward()* (this defines the forward computation of the model)

### PyTorch training loop
1. Pass the data through the model for a number of *epochs* (e.g. 100 for 100 passes of the data)
2. Pass the data through the mode, this will perform the *forward()* method located within the model object
3. *Calculate the loss value* (how wrong the model's predictions are)
4. *Zero the optimizer gradients* (they accumulate every epoch, zero them to start fresh each forward pass)
5. Perform *backpropagation* on the loss function (compute the gradient of every parameter with requires_grad=True)
6. *Step the optimizer* to update the model's parameters with respect to the gradients calculated by *loss.backward()*

> [!NOTE]
> all of this can be turned into a function
