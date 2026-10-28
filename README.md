# Convolutional Neural Networks

A convolutional neural network keeps an image as a grid and learns small filters that detect edges, strokes and shapes. In these notebooks you compute convolution and pooling with NumPy, then build and inspect a CNN with Keras. A final exercise applies them to images of clothing.

## Learning Objectives

By the end of this repository, you should be able to:

- Calculate the convolution of an image with a filter and the output size for a given padding and stride.
- Apply edge-detection filters and max pooling to an image with NumPy.
- Build a CNN in Keras with `Conv2D`, `MaxPooling2D`, `Flatten` and `Dense` layers, and calculate the output shape and parameters of each layer.
- Compare a CNN with a dense network by their validation accuracy and number of parameters.
- Plot the learnt filters and feature maps of a convolutional layer.
- Reduce overfitting in a CNN with dropout and early stopping, and evaluate it once on the test set.

## Learning Path

Work through the notebooks in this order:

| File / Folder | Description |
|---|---|
| [**01 - Convolution and Pooling**](01_convolution_and_pooling.ipynb) | Compute convolution, padding, stride and pooling with NumPy. |
| [**02 - A Convolutional Network with Keras**](02_cnn_with_keras.ipynb) | Build LeNet-5 for handwritten digits and inspect its filters. |
| [**03 - Exercise: Classify Clothing Images with a CNN**](03_exercise_fashion_mnist.ipynb) | Build, regularise and test a CNN on Fashion-MNIST. |

### Additional Folders and Files

| File / Folder | Description |
|---|---|
| [**Solutions**](solutions/) | Worked solutions for all three notebooks. |
| [**Assets**](assets/) | The figures used in the notebooks and the scripts that draw them. |
| [**pyproject.toml**](pyproject.toml) | Project configuration and dependencies. |
| [**uv.lock**](uv.lock) | Dependency lock file. |

## Setup

> [!NOTE]
> Throughout these steps, text in angle brackets like `<repo-name>` is a
> **placeholder**. Replace it, including the `< >` brackets, with your own
> value. For example, `cd <repo-name>` becomes `cd amle-convolutional-neural-networks`.

### 1. Create the Repository from the Template

Click **Use this template** on GitHub.

When creating the repository:

- Set yourself as the **Owner**
- Choose a repository name
- Disable **Include all branches**
- Click **Create repository**

> [!IMPORTANT]
> If you are working in pairs or groups, only **one person** should complete this step.

---

### 2. Add Collaborators (Pairs/Groups Only)

If working with teammates:

1. Open the repository on GitHub
2. Go to **Settings → Collaborators**
3. Add your teammates as collaborators
4. Share the repository link with your team

Teammates should accept the invitation before continuing.

---

### 3. Clone the Repository

Copy the SSH URL from the **Code** button on GitHub, then run:

```bash
git clone <copied-ssh-url>
```

The copied SSH URL will look like `git@github.com:<your-username>/<repo-name>.git`.

---

### 4. Move into the Project Folder and Install Dependencies

This installs all dependencies, including TensorFlow and Keras, and creates a virtual environment in `.venv/`. TensorFlow is a large package, so the first install takes a few minutes.

```bash
cd <repo-name>
uv sync
```

---

### 5. Open the Notebooks

> [!NOTE]
> Make sure you open VS Code from the project root so it automatically detects the environment created by `uv sync`.

Launch VS Code in the project root folder:

```bash
code .
```

Then open a notebook and select the Python environment created by `uv sync` as the kernel.

## References & Further Reading

- [**CS231n: Convolutional Neural Networks**](https://cs231n.github.io/convolutional-networks/): How convolutional and pooling layers work, and how their sizes and parameters are calculated.
- [**A guide to convolution arithmetic for deep learning**](https://arxiv.org/abs/1603.07285): Padding, stride and output sizes, explained with figures.
- [**Image Kernels explained visually**](https://setosa.io/ev/image-kernels/): Try blur, sharpen and edge filters on a photo in the browser.
- [**An Interactive Node-Link Visualization of Convolutional Neural Networks**](https://adamharley.com/nn_vis/cnn/2d.html): Draw a digit and watch every layer of a CNN respond.
- [**Deep Learning with Python**](https://deeplearningwithpython.io/): François Chollet's book on Keras, free to read online; chapter 8 covers image classification.
- [**TensorFlow tutorial: Convolutional Neural Network**](https://www.tensorflow.org/tutorials/images/cnn): A CNN on colour images, step by step.
- [**Fashion-MNIST**](https://github.com/zalandoresearch/fashion-mnist): The dataset of the exercise, with a benchmark of many models.
- [**CNN Explainer**](https://poloclub.github.io/cnn-explainer/): An interactive visualisation of a CNN on real photos.
