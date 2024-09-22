# CGI-ML-project
#### Authors: Shahr Samur & Dvir Adler

Our project aims to detect bone fractures by combining Ghost Imaging (GI) with machine learning. In this research, we explore the potential of using GI-captured X-rays and advanced algorithms to identify fractures more accurately. By using this technique, we hope to reduce radiation exposure during medical diagnoses, providing a safer alternative for patients undergoing X-ray evaluations. Our goal is to investigate whether this approach can enhance diagnostic precision while minimizing health risks.
This project is a continuing of a previus reaserch done by Noa Tal & Hadar Leiman, a documenatation of their project can be found [here](https://github.com/HadarLeiman/GI_Machine_Learning_Project/tree/master).


## Dataset
In our project we mainly used two Datasets:
* MNIST
* Wrist

The wrist dataset is a publicly available dataset of wrist X-ray images which we later preprocessed to generate GI measurements for training and evaluating our neural network model. You can download the dataset from [here](https://www.nature.com/articles/s41597-022-01328-z#Sec9).

## Code Parts
Our project consists of three code parts, each contained in its respective folder:

MINST: This folder contains the code for reproducing the results of this article on MNIST datset.

Wrist_transfer_learning: This folder holds the code where we applied transfer learning to the wrist fracture dataset.

Statistical_Analysis: This golder contain a code that perform analysis to the samples the model from the previous part didn't learn correctly.

To run the project please download/clone the folder you are interested in and see the following detailes.

