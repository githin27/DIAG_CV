# Transformer in Anomaly Detection: CNN vs GAN Approaches


# Abstract

<p style="text-align: justify;">
This repository contains code for evaluating hybrid neural network architectures combining Transformers with traditional neural network models—specifically Convolutional Neural Networks (CNNs) and Generative Adversarial Networks (GANs)—for anomaly detection tasks in computer vision.
The primary goal is to assess and compare the performance of two distinct hybrid architectures:

- **CNN + Transformer**
- **GAN + Transformer**

By integrating the strengths of CNNs (feature extraction), GANs (data augmentation and anomaly generation), and Transformers (contextual representation and attention mechanisms), these models aim to achieve improved accuracy and robustness in detecting anomalies.
</p>

Clone the project

```bash
  git clone https://github.com/githin27/DIAG_CV
```

# Set up docker
1. Make sure `Docker` and `Docker Compose` are installed on your system. 
2. Use docker composer to build and start the conatiner
  ```bash
    docker compose build
    docker compose up
  ```
3. for accessing the container shell 
  ```bash
    docker exec -it cv bash
  ```
4. Access Jupyter Lab by open the browser and navigate to:
  ```bash
    http://localhost:8888
  ```
The Dockerfile included in this repository leverages the official PyTorch image with CUDA support (pytorch/pytorch:latest) and includes:

- **PyTorch environment with CUDA support**
- **Dependencies installation via `requirements.txt`**
- **Preconfigured Jupyter Lab environment**


## Directory Structure

```
.
├── CV_1921463
    ├── workspace_sys
    |   ├── MvTec_dataset                   # sample dataset
    |   ├── CNN_trans
    |   │   ├── train.py
    |   │   ├── test.py
    |   │   ├── utils.py
    |   │   ├── default_dataset             # sample test data
    |   │   ├── resnet50_weights.pth
    |   ├── GAN_trans
    |   │   ├── train.py
    |   │   ├── test.py
    |   │   ├── utils.py
    |   │   ├── default_dataset              # sample test data
    ├── Dockerfile
    ├── docker-compose.taml
    ├── requirements.txt

```

# CNN + Tranformer
To train the code either can pass the dataset and the number of epochs:
```bash
  python3 cnn-_trans_train.py --dataset_path <path_to_tain_data> --num_epochs <num_of_epochs>
```
or just call the function which will run on default values for sample train data and number of eopchs:
```bash
  python3 cnn_trans_train.py 
```
For testing the trained modeleither you can pass the trained model and the test dataset:
```bash
  python3 cnn_trans_test.py --model_path <path_to_tained_model> --dataset_path <path_to_tain_data>
```
or just call the function which will run on thge defaulte pretrained model and the test data:

```bash
  python3 cnn_trans_test.py 
```
In all the cases, an output folder is created with time stamp which stores all the output of the code.

# GAN + Tranformer
To Train the code either can pass the dataset and the number of epochs:
```bash
  python3 gan_trans_train.py --dataset_path <path_to_tain_data> --num_epochs <num_of_epochs>
```
or just call the function which will run on default values for sample train data and number of eopchs:
```bash
  python3 gan_trans_train.py 
```
For testing the trained modeleither you can pass the trained model and the test dataset:
```bash
  python3 gan_trans_test.py --model_path <path_to_tained_model> --dataset_path <path_to_tain_data>
```
or just call the function which will run on thge defaulte pretrained model and the test data:

```bash
  python3 gan_trans_test.py 
```
In all the cases, an output folder is created with time stamp which stores all the output of the code.





