# NucEnhancedApp Enhance the Nucleus Images in live-cell embryos via Effective Deep Learning Model

## 1. Programme Introduction

This is a programme for running the fluorescence images of cell nucleus enhancement, especially for *C. elegans*. For other animals, you need to re-train the DNN model with about 4 embryos Ground Truth.

## 2. Running Steps

Make sure you have installed python3 (https://www.python.org/downloads/) and conda (https://docs.anaconda.com/free/miniconda/index.html) in your computer linux/osx/windows.

Then run the following command one by one in your terminal/command line/power shell.

**The first time**
* conda create --name EmbNucEnhancementPYEnvironment20240329Ver python=3.9.19
* conda activate EmbNucEnhancementPYEnvironment20240329Ver
* pip install PyQt5
* pip install stardist
* pip install nibabel
* pip install tensorflow
* python NucEnhanceApp.py

**Second and future**
* conda activate EmbNucEnhancementPYEnvironment20240329Ver
* python NucEnhanceApp.py

Then you get the ![APP](./static/document_imgs/app%20demo%20figure.png)

*Close the terminal directly*

## 3. Result comparison
The programme would enhance all recognized nucleus images and allow starrynite and acetree to trace the whole live-cell lineage ![as below](./static/document_imgs/result%20show%20figure.png).

The enhanced performance would increase as the cell number increase (The messier the image, the better enhancement).