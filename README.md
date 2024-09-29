# Medical diagnosis using X-ray imaging based on machine learning
#### Authors: Shahr Samur & Dvir Adler

Our project aims to detect bone fractures by combining Ghost Imaging (GI) with machine learning. In this research, we explore the potential of using GI-captured X-rays and advanced algorithms to identify fractures more accurately. By using this technique, we hope to reduce radiation exposure during medical diagnoses, providing a safer alternative for patients undergoing X-ray evaluations. Our goal is to investigate whether this approach can enhance diagnostic precision while minimizing health risks.
This project is a continuing of a previus reaserch done by Noa Tal & Hadar Leiman, a documenatation of their project can be found [here](https://github.com/HadarLeiman/GI_Machine_Learning_Project/tree/master).

## Rquierments
Install the required packages using the command:
`pip install torch torchvision pandas numpy tqdm Pillow wandb`

## Dataset
In our project we mainly used two Datasets:
* MNIST
* Wrist

The wrist dataset is a publicly available dataset of wrist X-ray images which we later preprocessed to generate GI measurements for training and evaluating our neural network model. You can download the dataset from [here](https://www.nature.com/articles/s41597-022-01328-z#Sec9).

## Code Parts
Our project consists of three code parts, each contained in its respective folder:

MINST: This folder contains the code for reproducing the results of this article on MNIST datset.

Wrist_transfer_learning: This folder holds the code where we applied transfer learning to the wrist fracture dataset.

Statistical_Analysis: This folder contain a code that perform analysis to the samples the model from the previous part didn't learn correctly.

Processed_Dataset: This folder contains the datset after preprucessing. There are two datasets in the folder, a regulat one called 'new_dataset_1024_64_128.csv` and one after augmentation called `augmantation_data.csv`

To run the project please download/clone the folder you are interested in and see the following detailes.

## How to use
The first part is independent, independent the other parts come together. The results of the second part are necessary to run the third part.
### MNIST
There are two files in this folder - `Gi_Minst_Regular.py` which is a neural netwotk that we built using the packgae `keras`, and `GI_Minst_research_model.py` which is a neural network based on the architecture from [this article](https://pubmed.ncbi.nlm.nih.gov/34624000/). To run this part follow these step:
1. Download the MNIST folder
2. Choose a neural network you wnat and run the choosen `.py` file

Note: in `Gi_Minst_Regular.py` you should define a path to save the trained model for future use.

### Wrist_transfer_learning + Statistical_Analysis

Before running, to define which dataset to use go to `model_pipeline.py` and in the `make_loaders` function choose between the regular dataset to the augmemted dataset.

To run this part downlaod the `Wrist_transfer_learning` and locate the main file - `run.py` and run it.

In the `model_pipline.py` and `our_nn.py` files you can adjust the model configuration and hyper parameters such as learning rate, number of epochs etc. In addition you can switch between few networks architectures like efficientnet, ResNet152 or VGG.

The output of this code section is the acuuracy of the trained model and two files - `false_negatives.csv` and `false_positives.csv`. In this files there are samples that recognized incorrectly by the model and classsified to the wrong output class. These file are used for downstream analisys.

To get a bird's eye view about the samples that the model didn't learn correctly you can downlaod the `Statistical_Analysis folder`. The folder contains a `stat.py` file that gives statistical analysis and insights about the samples, and three `.csv` files which are the input files foe the code. All the `.csv` files should be place in the same directory as `stat.py`. The file `dataset.csv` is givven in this folder, and for `false_negatives.csv` and `false_positives.csv` you can either use the output of the previus section or use the givven files. 


### Run with Wandb
We provide the ability to run experiments using Weights and Biases (wandb) for tracking hyperparameter sweeps and logging results. To use this feature, follow these steps:

1. Setup wandb:  
Make sure you have a wandb account and have installed the wandb library:
`pip install wandb`

2. Login to wandb:
Before running the code, log in to your wandb account using the following command:
`wandb login`

3. Run the sweep:  
Go to the Wrist_transfer_learning folder.
Run the `main_sweep.py` file to initiate the wandb sweep:
`python main_sweep.py`

4. Configuration: 
The `main_sweep.py` file handles the configuration for the wandb sweep. It will use the model_pipeline_sweep.py file for training and evaluating the model.
You can customize hyperparameters such as learning rate, number of epochs, model architecture, and more within the`main_sweep.py` file.

5. Dataset Choice:
You can switch between the original preprocessed dataset or the augmented dataset by modifying the paths in `model_pipeline_sweep.py`.
Both datasets are located in the Processed_Dataset folder.






