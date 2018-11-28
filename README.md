# CS131: Computer Vision Foundations and Applications
This repository contains the released assignments for the [fall 2017](http://vision.stanford.edu/teaching/cs131_fall1718/) and [fall 2018](http://vision.stanford.edu/teaching/cs131_fall1819) iteration of CS131, a course at Stanford taught by [Juan Carlos Niebles](http://www.niebles.net) and [Ranjay Krishna](http://ranjaykrishna.com).

The assignments cover a wide range of topics in computer vision and should expose students to a broad range of concepts and applications. Homework 0 sets up the course with an introduction on how to use images in python and numpy. It covers basic linear algebra that would be helpful through the course.

Homework 1 starts off the topics in computer vision by understanding concepts such as convolutions, linear systems and different kernels and how to design them to find certains signals in images. Homework 2 focuses on edge detection, applying it to lane detection to aid self driving cars. Homework 3 introduces SIFT and RANSAC, which are useful for finding corresponding points in multiple images, enabling applications like panorama creation, which is a common feature in most of our smart phones.

Moving beyond pixels and edges, homework 4 takes a wider view of the image and asks students to use a dynamic programming algorithm to define the energy of certain regions in an image. This energy definition allows us to find regions that are important, enabling different monitors with varying sizes (large projections, medium laptop screens and small cell phones) to display only the important portions of an image. We specifically ask students to implements different versions of seam carving while preserving the structure of objects in images.

Homework 5 asks students to learn to segment images by assigning each pixel to a cluster, where each cluster represents a semantic object category. We analysze different unsupervised clustering algorithms like K-means and hierarchical aggolomerate clustering techniques. They also learn to apply their techniques to segments cats from images and evaluate how well different methods perform. 

Homework 6 and 7 introduce two fundamental tasks in computer vision: image classification and object detection respectively. Students learn to use different image features and classifiers to study how they influence results on such tasks. They are asked to consider the important of rotation, translation, scale and occlusion variance when deciding what features to use. They also study the curse of dimensionality and learn to apply principal component analysis and linear discriminant analysis to further improve their features while compressing their features. Students also explore common methods like sliding windows and deformable parts models to decompose objects into its individual components when detecting them.

Homework 8 ends the course by moving away from still images and into video and studying time dependent features. They learn to use optical flow to calculate motion in images, a technique useful for tracking people in images and classifying actions that people perform.

All the homeworks are under the MIT license.
