# VICReg-model
VICReg model project

## Introduction

This repository presents VICReg, a self-supervised learning model that learns meaningful representations from images.
VICReg is designed to create powerful visual representations that can be useful for various downstream tasks. 
It's composed of an encoder and a projection head (which is omitted at test time). 

## Key Concepts

VICReg optimizes 3 different objectives using the projections z and z′ of each image in its batch(where z, z′ - two different augmentations of the same image).
We name the 3 objectives as the Invariance, Variance and Covariance objectives:

Invariance Objective - This objective drives the encoder to be invariant to the performed augmentations (T)

Variance Objective. The variance objective forces each embedding vector (projection) in the batch to be different, by making sure each dimension in the representation is meaningful.
This is performed by optimizing the standard deviation of each dimension to be above a known threshold, using the hinge loss.

Covariance Objective. This objective aims to decorrelate the variables of the embedding, thereby preventing it to collapse to only a few dimensions. 
It does so, by minimizing the covariance between the different dimensions of the embedding.




