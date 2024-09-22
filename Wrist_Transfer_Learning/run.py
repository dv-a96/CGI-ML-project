# this file is to call the model_pipeline function with the config object
from model_pipeline import model_pipeline
from collections import namedtuple

# define the configuration of the model
Config = namedtuple('Config',
                    ['num_of_layers_unfreeze', 'model_name', 'lr', 'epochs', 'reshape_size', 'num_of_measurements'])

# main function to call the model_pipeline function with the config object
if __name__ == '__main__':
    # define the configuration of the model
    config = Config(num_of_layers_unfreeze=2,
                    model_name='ResNet152',
                    lr=0.001,
                    epochs=5,
                    reshape_size=(64, 128),
                    num_of_measurements=1024)
    # call the model_pipeline function with the config object
    print("config done")
    accuracy = model_pipeline(config)
    print("modle is redy")
    print("accuracy:", accuracy)
