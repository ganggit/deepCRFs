# deepCRFs

# Introduction
This code implements the deep CRFs for sequential labeling. Basically, it pretrains the deep neural network to initialize 
the all weights in an independent manner (no correlation considered, but it helps to initialize the whole structure and weights)
Then, we use online learning to update all weights via backpropagation. In the top layer, we use perception learning and the lower
level layer weights are updated with backpropagation. One vital step to make the whole model generalized well is to reinitialize
the top layer weight.


# Functionality
The code can be used for any sequential labeling problem, such as POS tagging, handwritten recogniton and so on.


# Demo
deep_ocr_experiment


# Reference
Sequential Labeling with online Deep Learning, in ECML 2016
Authors: Gang Chen , Ran Xu and Sargur Srihari, SUNY at Buffalo
Email: gangchen@buffalo.edu
