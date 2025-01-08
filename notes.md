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
