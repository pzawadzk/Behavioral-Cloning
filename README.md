# behavioral-cloning

This project implements a machine learning model for self-driving car in a simulator. The model uses camera images of the road and predicts appropriate steering angle.  The overall approach is to use bahavioural cloning to trained the model.
The first step is to create a set of "good behaviours" that we want the model to reproduce. In the context of self-driving car the most importnat good behaviour is driving in the center of the road.  Second we also need to prevent driving off the road.


take a sereis of camara images and corepsdodning steering angles to train a convlutional nuearla netowerk. The trained model then can be used to generate new steering  angles. 

 and teach car how to recover from prevent "bad bahvaiours" that is driving off the road;

Thus to create a training set for the model we drive the car in the simulator trying to keep the car in the center of the road and collect camara images and correpsonding steering angles. 

1. Training set:
-  Driving car in a normall fashiotn that is tryting to keep it in the center of rooed. 
- 
- 


There Is the solution design documented?

The README thoroughly discusses the approach taken for deriving and designing a model architecture fit for solving the given problem.

# Model architecture:

- Layer 1:
Convolutional (Input = 32x32x1. Output = 28x28x10)
Relu Activation
Max Pooling (Input = 28x28x10. Output = 14x14x10)

- Layer 2:
Convolutional (Input = 14x14x10. Output = 10x10x24)
Relu Activation
Max Pooling (Input = 10x10x24. Output = 5x5x24)

- Layer 3:
Fully Connected (Input = 600. Output = 200)
Relu Activation
Layer 4:
Fully Connected (Input = 200. Output = 100)
Relu Activation
Layer 5:
Fully Connected (Input = 100. Output = 43)



Is the model architecture documented?

The README provides sufficient details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.

Is the creation of the training dataset and training process documented?

The README describes how the model was trained and what the characteristics of the dataset are. Information such as how the dataset was generated and examples of images from the dataset should be included.
