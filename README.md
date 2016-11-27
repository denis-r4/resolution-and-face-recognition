# resolution-and-face-recognition

## Index

* [hypothesis](#hypothesis)
* [hypothesis-testing-plan](#hypothesis-testing-plan)
* [data-acquisition-and-preparation](#data-acquisition-and-preparation)
* [face-recognition-model](#face-recognition-model)
* [test](#tests)
* [results-analysis](#results-analysis)

## hypothesis
Face recognition systems got widespread in problems of a processing and analysis of a video data. One of them is the problem of home security and an example of product based on a home security is ["Ring.com"](https://ring.com) company.

At the moment, there are many solutions of face recognition systems, that have accuracy close to 99% according to [LFW-behcnmark](http://vis-www.cs.umass.edu/lfw/results.html)

However, detection systems based on a CNN imposes various restrictions on an input data. One of them - spatial resolution of an input image (as a rule, the height, and width of an image) must be equal to the size of a network input. In practice, there are often cases when the target object in a frame has a smaller size than the size of the entrance window of the network, which may a negatively affects the quality of facial recognition.

<br/>
<p align="center">
  <img src="https://github.com/denis-r4/resolution-and-face-recognition/blob/master/media/hypothesis.png">
  <img src="https://github.com/denis-r4/resolution-and-face-recognition/blob/master/media/face_3.png">
</p>

A particular solution of this problem is an artificial increasing of a resolution of the input image up to the size of the input of the model. However, despite the fact that this problem has been solved with good precision, when the system is applied to a large amount of data (I mean Big Data) the profit in tenths and hundredths of a percent of accuracy may bring a significant contribution.

One way to achieve a small growth of a recognition quality under the conditions described above is the using of "more intelligent" techniques to improve image resolution. In this research, I will test how the standard interpolation techniques (most spread), as well as algorithms of super-resolution, can be applied to the target image resolution, and affect the final quality of facial recognition.  

|   source (50x50)   |   nearest x4  |   bilinear x4   |   bicubic x4   |   super-resolution x4   |
|   -------------------   |:------:|:------:|:------:|:------:|
|   <p align="center"> <img src="https://github.com/denis-r4/resolution-and-face-recognition/blob/master/media/resize_examples/source_50x50.png"> </p>   |   <p align="center"> <img src="https://github.com/denis-r4/resolution-and-face-recognition/blob/master/media/resize_examples/nearest.png"> </p>   |   <p align="center"> <img src="https://github.com/denis-r4/resolution-and-face-recognition/blob/master/media/resize_examples/bilinear.png"> </p>   |   <p align="center"> <img src="https://github.com/denis-r4/resolution-and-face-recognition/blob/master/media/resize_examples/cubic.png"> </p>   |   <p align="center"> <img src="https://github.com/denis-r4/resolution-and-face-recognition/blob/master/media/resize_examples/super-resolution.png"> </p>   |



## hypothesis-testing-plan
To test the hypothesis, I will need an in-the-wild faces data, a model for face recognition, image resolution enhancement methods, and python.
In order to understand how different methods of image resolution enhancement affect the quality of face recognition, I will exploit the next pipeline:

* Detect [faces-ROI](https://github.com/denis-r4/resolution-and-face-recognition/blob/master/notebooks/face_detector.ipynb) on real data, as well as artificially noise them, reduce the size of ROI faces, simulating different variants of object distance from the camera;
* Compute vector representation of faces for the database and for all generated on the stage one samples using [pre-trained](http://www.robots.ox.ac.uk/~vgg/software/vgg_face) face recognition model [VGG16](https://github.com/denis-r4/resolution-and-face-recognition/blob/master/notebooks/vgg.py) using last before output FC layer of the network;
* Compute [cosine distance](https://github.com/denis-r4/resolution-and-face-recognition/blob/master/notebooks/compute_cos_distance.ipynb) from each input sample (from the stage two)  to each hash in the face database;
* Compute a network error for various boundary condition of faces and different methods of a resolution enhancement;
* Analysing the recognition error from method to method.


## data-acquisition-and-preparation

## face-recognition-model

## tests

## results-analysis
