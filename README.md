# deepCRFs
Deep conditional random fields for sequential labeling

# Introduction
This code implements the deep CRFs for sequential labeling. Basically, it pretrains the deep neural network to initialize 
the all weights in an independent manner (no correlation considered, but it helps to initialize the whole structure and weights)
Then, we use online learning to update all weights via backpropagation. In the top layer, we use perception learning and the lower
level layer weights are updated with backpropagation. One vital step to make the whole model generalized well is to reinitialize
the top layer weight.

# Functionality
The code can be used for any sequential labeling problem, such as POS tagging, handwritten recogniton and so on.
(1) learn_deepneuralnetwork.m, it will learn the deep model in an independent manner, this is from Hinton

(2) deep_crf_2nd_online.m, it will learn the deep CRFs model in the paper, which will reinitialize the top layer weight
and update lower level weights in an online manner.

# Demo
deep_ocr_experiment


# Reference
Sequential Labeling with online Deep Learning, in ECML 2016 

Authors: Gang Chen , Ran Xu and Sargur Srihari, SUNY at Buffalo

Email: gangchen@buffalo.edu
