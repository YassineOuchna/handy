# handy
Implementing and training a computer vision AI that detects a human hand. \
This project is heavily inspired by the following [article](https://towardsdatascience.com/gentle-introduction-to-2d-hand-pose-estimation-approach-explained-4348d6d79b11).

## Raw dataset 
To train our model, we first need a lot of pictures of hands with different backgrounds and their respective labels. \
The dataset (download link [here](https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip)) contains two batches of 130 240 pictures of hands to train and to evaluate.
Each batch of images has its own batch of corresponding labels as well as coordinates of the camera position, useful to map the 3D coordinates to 2D. The dataset has been augmented with a factor of 4, meaning the true raw images consist of just 32560 = 130240/4 files with corresponding 32560 labels. \

Here are some of the outputs of the file *data_processing.py* : 

*Images with labels* 

<img src="./demos/images_labels.png?raw=true" />

*Corresponding layers*

<img src="./demos/layers.png?raw=true" />

## Model architechture 
The neural network model used in the article is beautifully described by the following illustration : 

*U-net model originally used in [article](https://towardsdatascience.com/gentle-introduction-to-2d-hand-pose-estimation-approach-explained-4348d6d79b11)*

<img src="./demos/model.png?raw=true" />

I'm currently playing around with different parameters and testing some layers such as SpatialDropOut on top of this model. \
However modifying and testing on such a huge dataset takes a lot of time even when using top tier GPUs in GoogleCollab, one epoch with a batch size of 64 takes roughly 2 hours! 



