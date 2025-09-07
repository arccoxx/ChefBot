# ChefBot: Advanced Deep Belief Network Training on Recipe1M+

This repository contains a PyTorch implementation for training a Deep Belief Network (DBN) on the large-scale Recipe1M+ dataset. The project serves as a proof-of-concept for advanced, physically-inspired training techniques for deep generative models.

The core of this project is a two-phase training pipeline that first learns the underlying structure of the recipe data in an unsupervised manner and then fine-tunes the learned representations for a specific classification task. It leverages a robust, ACA-like (Adaptive Cluster Annealing) solver for pre-training and uses Annealed Importance Sampling (AIS) to intelligently guide the training process.

## Core Features

*   **Deep Belief Network Architecture:** The network architecture is programmatically determined to create a deep "funnel" of layers, forcing the model to learn progressively more abstract features of the data.
*   **ACA-like Pre-training:** Each layer of the DBN is pre-trained as a Restricted Boltzmann Machine (RBM) using a highly parallelized Parallel Tempering solver, which is more powerful than standard Gibbs sampling.
*   **AIS-Guided Early Stopping:** The number of training epochs for each layer is not fixed. After each epoch, the model's fit is evaluated on a validation set by estimating the log-likelihood via Annealed Importance Sampling (AIS). Training for a layer stops automatically when its performance on the validation set no longer improves, preventing overfitting and saving significant time.
*   **Supervised Fine-tuning:** After generative pre-training, a classification head is added, and the entire network is fine-tuned with standard backpropagation to predict the presence of a target ingredient.
*   **GPU Acceleration:** The entire pipeline is built in PyTorch and is heavily optimized to run on CUDA-enabled GPUs.
*   **Smoke Test Mode:** The script includes a built-in smoke test mode (`SMOKE_TEST = True`) that runs the entire pipeline on a small subset of the data with a smaller model, allowing for rapid verification of correctness and reliability before committing to a full, resource-intensive training run.

## How It Works: The Training Pipeline

The script automates a complete, multi-stage training and evaluation process.

1.  **Data Processing:** The script automatically downloads the Recipe1M+ dataset. It then creates custom PyTorch `Dataset` objects to serve multi-hot encoded recipe vectors.
2.  **Train/Validation/Test Split:** The data is split into three sets: a training set for model updates, a validation set for the early stopping mechanism, and a final test set for evaluating the fine-tuned model.
3.  **Phase 1: Unsupervised Pre-training:**
    *   For each layer in the DBN architecture:
        *   The RBM is trained on the output of the previous layer using the Parallel Tempering solver.
        *   After each epoch, the RBM's log-likelihood is estimated on the validation set using ACA-enhanced AIS.
        *   If the log-likelihood fails to improve for a set number of "patience" epochs, training for this layer stops, and the best-performing weights are saved.
4.  **Phase 2: Supervised Fine-tuning:**
    *   The pre-trained RBM layers are "unrolled" to form a deep feed-forward neural network.
    *   A final classification layer is added.
    *   The entire network is trained using the Adam optimizer and backpropagation on a specific task (e.g., predicting the presence of 'salt').
5.  **Model Saving:** The final, fine-tuned classifier's weights (`state_dict`) are saved to a file named `dbn_finetuned_classifier.pth`.

## Getting Started

### Prerequisites

You will need a CUDA-enabled GPU and the following Python libraries installed.

```bash
pip install torch torchvision tqdm numpy scikit-learn
```

### How to Run

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/arccoxx/AdaptiveClusterAnnealing.git
    cd AdaptiveClusterAnnealing
    ```

2.  **Run the Smoke Test (Recommended First Step):**
    To ensure everything is working correctly on your machine without waiting for hours, you can run the built-in smoke test. Set the flag at the top of the script:
    ```python
    SMOKE_TEST = True
    ```
    Then, run the script. This will train a smaller DBN on a tiny subset of the data for only a few epochs but will execute every step of the pipeline, including the reliability checks.

3.  **Run the Full Training:**
    Once the smoke test passes, you can launch the full training run. Set the flag to `False`:
    ```python
    SMOKE_TEST = False
    ```
    Then, run the script.

    **Warning:** This is a computationally intensive process that will take a significant amount of time and GPU resources.

### Output

After the script finishes, you will have two primary outputs:

1.  **Console Logs:** Detailed logs showing the progress of each pre-training epoch, the validation log-likelihood at each step, and the final test accuracy of the fine-tuned classifier.
2.  **Saved Model:** A file named `dbn_finetuned_classifier.pth` containing the weights of your fully trained model, ready to be loaded for inference.
