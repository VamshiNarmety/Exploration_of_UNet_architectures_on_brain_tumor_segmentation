# Exploration of UNet architectures on brain tumor segmentation
This project implements and compares multiple UNet-based architectures (UNet, UNet++, Attention UNet, and Sharp UNet) for brain tumor segmentation using MRI images.  
Performance is evaluated using metrics like Dice, IoU, Precision, and Recall, with experiments conducted across different encoder backbones.

#Dataset

The dataset used is the **Brain Tumor MRI Segmentation** dataset. Make sure the dataset is downloaded and placed appropriately before training or evaluation. The structure and path may need to be adjusted depending on your platform (Kaggle, Colab, or local machine).

 Dataset used: [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

# setup the project
The code was run on Ubuntu with the following setup:

1. Install Python 3.10.x

2. From the terminal, run the following:

   ```bash
   pip3 install -r requirements.txt
and also code was run on kaggle notebooks for accessing GPUs, so make sure to change the paths for using any file in the code depending on whether you are running the code on your local machine or any cloud services like google colab/kaggle.

