# handy
Implementing and training a computer vision AI that detects a human hand. \
This project is inspired by the following [article](https://towardsdatascience.com/gentle-introduction-to-2d-hand-pose-estimation-approach-explained-4348d6d79b11).


## Raw dataset 
To train our model, we first need a lot of pictures of hands with different backgrounds and their respective labels. \
The dataset (download link [here](https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip)) contains two batches of 130 240 pictures of hands to train and to evaluate.
Each batch of images has its own batch of corresponding labels as well as coordinates of the camera position, useful to map the 3D coordinates to 2D. The dataset has been augmented with a factor of 4, meaning the true raw images consist of just 32560 = 130240 / 4 images with corresponding 32560 labels.

Here are some of the outputs of the file *data_processing.py* : 

*Images with labels* 

<img src="./demos/images_labels.png?raw=true" />

*Corresponding layers*

<img src="./demos/layers.png?raw=true" />

## Model architechture 
The neural network model used in the article is beautifully described by the following illustration : 

*U-net model originally used in [article](https://towardsdatascience.com/gentle-introduction-to-2d-hand-pose-estimation-approach-explained-4348d6d79b11)*

<img src="./demos/model.png?raw=true" />

*Model implementation with a batch size of 32 using **Tensorflow***

<img src="./demos/model_summary.png?raw=true">

# Try it yourself
## Environments 
Python == 3.9.18, Tensorlfow == 2.10, Cuda == 12.3, matplotlib, numpy, json, openCV

## Setup
Downloading the dataset & setting up the repo :
```bash
git clone https://github.com/YassineOuchna/handy.git
cd handy/data
wget https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip
unzip FreiHAND_pub_v2.zip
```
## Data processing and visualization
To process the data and visualize the first layer :
```bash
python data_processing.py
```
## Training
You can process the data and train the model on it directly by running :
```bash
python cnn_model.py
```
The model is then saved as `.keras` zip archive
## Testing on dataset images
You an visualize the output of the model on some images by running :
```bash
python test.py
```
## Real-time testing
You can use your own camera and visualize the output of the model in real time (by predicting each frame) by running :
```bash
python hand_detector.py
```



