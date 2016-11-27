# resolution-and-face-recognition

## Index

* [hypothesis](#hypothesis)
* [hypothesis-testing-plan](#hypothesis-testing-plan)
* [data-acquisition-and-preparation](#data-acquisition-and-preparation)
* [face-recognition-model](#face-recognition-model)
* [test](#tests)
* [results-analysis](#results-analysis)

## hypothesis
Face recognition systems got widespread in problems of a processing and analysis of a video data. One of them is the problem of home security and an example of product based on a home security is "Ring.com" company.

At the moment, there are many solutions of face recognition systems, that have accuracy close to 99% (according to vis-www.cs.umass.edu/lfw/results.html)

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


## data-acquisition-and-preparation

## face-recognition-model

## tests

## results-analysis
