# CIFAR-10 Classification Example

This example demonstrates how OpenEvolve can improve a basic image classification model on the CIFAR-10 dataset.

The initial program defines three functions:

- `load_data()` – downloads CIFAR-10 and returns training and test loaders.
- `init_model()` – creates a very small convolutional neural network.
- `train()` – trains the model and prints the loss and accuracy at each epoch.

The evaluator uses cascade evaluation. The first stage trains for a single epoch
with a short timeout. If the accuracy after the first epoch exceeds a threshold,
the second stage runs a longer training session.

Run the example with:

```bash
cd examples/cifar10_classification
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml
```
