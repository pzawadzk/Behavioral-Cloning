# Behavioral cloning

This project implements a machine learning model for a self-driving car in the Udacity simulator (Fig. 1). The model uses road images taken by three on-board cameras and predicts appropriate steering angle.  The model is trained using the so-called  behavioural cloning approach. 
In behavioural cloning we create a set of behaviours that we want the model to reproduce such as driving in the middle of the road.  In addition to reproducing "good" behaviours the model should also be able to recover from "bad behaviours" such driving on the shoulder.

<p>
<img src="simulator.png" width="480" alt="Combined Image" />
    <em>Fig. 1. View from the central camaare when car is on shoulder.</em>
</p>

## Model details

### Data collections

#### Normal driving

The most important behaviour for a self-driving car is to keep the car in the center of the road. 
To reproduce this behaviours we drive a car in the simulator keeping it in the center of the road and collect camera images and the corresponding steering angles. Total number of 2000 data points were generated.

#### Recovery from shoulder
In addition to recording "good behaviours" I also record behaviours necessary for the car to recover from the shoulder.
This is accomplished in three steps:

1. Drive the car to the shoulder 
2. Steer wheels toward the center of the road
3. Record camera images (Fig. 2) and steering angles of 25 ${\textdegree}$ for about 1 second at low speed (about 1 mph).

Total number of 300 data points were generated for left and right shoulders.
<p>
<img src="center_example.jpg" width="480" alt="Combined Image" /> <br>
    <em>Fig. 2. View from the central camaare when car is on shoulder.</em>
</p>

##### Synthetic data
To improve the recovery and smoothen the self-driving I supplement the recovery data set with synthetic data points.
The underling idea is that the closer the car is to the shoulder the larger steering angle (absolute value) needs to be for the car to recover.
One way to generate such data set would be repeat steps 1-3 with car slightly away from the shoulder and use a low steering angle.
Here, instead of collecting additional data I use left (right) camera image to approximate central camera image of car being close to right (left) shoulder.  Fig. 3 shows a view from the left camera when the car is on the right shoulder. After cropping the bottom of the image, the left camera image is a good approximation of the central camera image.

<p>
<img src="left_example.jpg" width="480" alt="Combined Image" /> <br>
    <em>Fig. 3. View from the central camaare when car is on shoulder.</em>
</p>

### Preprocessing


### Model architecture 

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
