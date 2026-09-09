# Fashion-MNIST Image Classification with CNN

A convolutional neural network built from scratch to classify grayscale clothing images into 10 categories, with a deep dive into where and why the model struggles.

## Dataset
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) — 70,000 grayscale images (28x28 pixels) across 10 clothing categories (T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot). Loaded directly via Keras, split into 60,000 training and 10,000 test images.

## Approach
1. **Preprocessing**: Normalized pixel values from a 0-255 range to 0-1, and reshaped images to include an explicit channel dimension `(28, 28, 1)` as required by Keras' Conv2D layers.
2. **Model Architecture**: Built a CNN with 2 Conv2D + MaxPooling2D blocks, followed by a Flatten layer and two Dense layers, ending in a 10-unit softmax output for multi-class classification.
3. **Training**: Trained for 4 epochs using the Adam optimizer and sparse categorical crossentropy loss (chosen because labels are integers, not one-hot encoded), with a 20% validation split to monitor generalization during training.
4. **Evaluation**: Assessed performance using training, validation, and test accuracy, followed by a full 10-class confusion matrix to identify specific misclassification patterns.
5. **Error Analysis**: Investigated the model's weakest category (shirts) by manually inspecting misclassified images.

## Results

| Metric | Accuracy |
|---|---|
| Training | 92.6% |
| Validation | 90.7% |
| **Test** | **90.6%** |

The close alignment between validation and test accuracy indicates the model generalized well, with no significant overfitting despite the short training run.

## Key Finding
The confusion matrix revealed that most categories (trousers, sandals, sneakers, bags) were classified with 97%+ accuracy, but **shirts were correctly classified only 71% of the time**, frequently confused with T-shirts, pullovers, and coats. Manually inspecting the misclassified images showed that this confusion is largely a **data limitation, not a model failure** — even a human observer would struggle to distinguish these categories at 28x28 grayscale resolution, since fine details like collar shape and fabric texture are lost. This suggests further accuracy gains would require higher-resolution input rather than additional model complexity or training time.

## Tools Used
Python, TensorFlow/Keras, NumPy, Matplotlib, Seaborn, Google Colab

## Future Improvements
- Train for more epochs with a GPU runtime to confirm the overfitting trend holds or stabilizes
- Add data augmentation (rotation, zoom, flip) to see if it improves shirt/pullover distinction
- Test with a higher-resolution version of similar clothing images to confirm the resolution hypothesis
- Add a third Conv2D block to test whether increased model depth captures finer distinguishing features

## How to Run
1. Open the notebook in Google Colab (no dataset download needed — Fashion-MNIST loads directly via Keras)
2. Run all cells in order
