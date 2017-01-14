# Behavioral cloning

This project implements machine learning model for a self-driving car in the UDACITY simulator (Fig. 1). The model uses road images taken by three on-board cameras to predict steering angle.  The model is trained using the so-called  behavioural cloning approach. 

<p>
<img src="simulator.png" width="480" alt="Combined Image" /> <br>
    <em>Fig. 1. Udacity self-driving car simulator.</em>
</p>

## Data
The goal of behavioural cloning is to reproduce human behaviours in a computer program. For a self-driving car the two most important behaviours are:

1. Normal driving i.e. driving in the middle of the lane;
2. Recovery from mistakes such as driving on the shoulder.

### Data collections

#### Normal driving
To recored normal driving behaviours I drive a car in the simulator keeping it in the center of the road and collect camera images and the corresponding steering angles.

<p>
<img src="normal_center_example.jpg" width="480" alt="Combined Image" /> <br>
    <em>Fig. 1. View from the central camera when the car is driving in the center the road.</em>
</p>

#### Recovery from shoulder
To record behaviours necessary for the car to recover from the shoulder I:

1. Drive the car to the shoulder; 
2. Steer wheels toward the center of the road;
3. Record camera images (Fig. 2) and steering angles of 25 ${\textdegree}$ for about 1 second at low speed (about 1 mph).

<p>
<img src="center_example.jpg" width="480" alt="Combined Image" /> <br>
    <em>Fig. 2. View from the central camera when the car is on the shoulder.</em>
</p>

##### Synthetic data
To improve the recovery and smoothen self-driving I supplement the recovery data set with synthetic data points.
The underling idea is that the further the car is from the shoulder the lower steering angle needs to be for the car to recover to the middle of the road.
One way to generate such data set would be repeat steps 1-3 with the car slightly away from the shoulder and use a lower steering angle.
Here, instead of collecting additional data, I use left (right) camera image to approximate central camera image when the is close to the right (left) shoulder.  Fig. 3 shows an example from the left camera when the car is on the right shoulder.

<p>
<img src="left_example.jpg" width="480" alt="Combined Image" /> <br>
    <em>Fig. 3. View from the left camara when the car is on the right shoulder.</em>
</p>

### Preprocessing
1. Normalize image data to [-0.5, 0.5] range
2. Resize from 320x160 px to 200x100 px
3. Crop to 200x60 px 

## Model architecture 

The model architecture was inspired by the LeNet architecture and the recommended [NVIDIA paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). A dropout was included in the convolutional layers to severe as a regularization parameter. The output layer is followed by hyperbolic tangent to limit steering angle predictions to [-1, 1] range.

* Layer 1: Input: 200x100x3, Output: 28x98x24, Convolutional: 5x5, Max Pooling: 2x2,  Dropout: 0.4,  Relu Activation
* Layer 2: Input: 28x98x24 Output: 12x47x36, Convolutional: 5x5, Max Pooling: 2x2,  Dropout: 0.4,  Relu Activation
* Layer 3: Input: 12x47x36, Output: 4x21x48, Convolutional: 5x5, Max Pooling: 2x2,  Dropout: 0.4,  Relu Activation
* Layer 4: Input: 4x21x48, Output: 2x19x6, Convolutional: 3x3, Max Pooling: 1x1,  Dropout: 0.4,  Relu Activationn
* Layer 5: Input: 2432, Output: 1000, Fully Connected: 1000 neurons, Relu Activation
* Layer 6: Input: 1000, Output: 100, Fully Connected: 100 neurons, Relu Activation
* Layer 7: Input: 100, Output: 20, Fully Connected: 20 neurons, Relu Activation
* Layer 8: Input: 20, Output: 1, Fully Connected: 1 neuron, Tanh Activation 

### Training procedure
The model was trained on AWS EC2 g2.2xlarge instance. Dropout rate was varied to ensure model does not overfit the data. Validation error converged after 5 epochs.

## Results
The model performs well in the simulator without crashing even after multiple loops.
