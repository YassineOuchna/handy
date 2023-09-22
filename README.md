# handy
Implementing and training a computer vision AI that detects a human hand. \
This project is heavily inspired by the following [article](https://towardsdatascience.com/gentle-introduction-to-2d-hand-pose-estimation-approach-explained-4348d6d79b11).

## Raw dataset 
To train our model, we first need a lot of pictures of hands with different backgrounds and their respective labels. \
The [dataset](https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip) contains two batches of 130 240 pictures of hands to train and to evaluate.
Each batch of images has its own batch of corresponding labels as well as coordinates of the camera position, useful to map the 3D coordinates to 2D. The dataset has been augmented with a factor of 4, meaning the true raw images consist of just 32560 = 130240/4 files with corresponding 32560 labels. 




