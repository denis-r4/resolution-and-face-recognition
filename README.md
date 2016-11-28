# resolution-and-face-recognition

## Index

* [hypothesis](#hypothesis)
* [hypothesis-testing-plan](#hypothesis-testing-plan)
* [data-acquisition-and-preparation](#data-acquisition-and-preparation)
* [face-recognition-model](#face-recognition-model)
* [test](#tests)
* [results-analysis](#results-analysis)
* [timing](#timing)

## hypothesis
Face recognition systems got widespread in problems of a processing and analysis of a video data. One of them is the problem of home security and an example of product based on a home security is ["Ring.com"](https://ring.com) company.

At the moment, there are many solutions of face recognition systems, that have accuracy close to 99% according to [LFW-behcnmark](http://vis-www.cs.umass.edu/lfw/results.html)

However, detection systems based on a CNN imposes various restrictions on an input data. One of them - spatial resolution of an input image (as a rule, the height, and width of an image) must be equal to the size of a network input. In practice, there are often cases when the target object in a frame has a smaller size than the size of the entrance window of the network, which may a negatively affects the quality of facial recognition, more than that, frequently, undersized or/and blurred face images just skipped.

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
* [Compute](https://github.com/denis-r4/resolution-and-face-recognition/blob/master/notebooks/compute_cos_distance.ipynb) [cosine distance](https://en.wikipedia.org/wiki/Cosine_similarity) from each input sample (from the stage two)  to each hash in the face database;
* Compute a network error for various boundary condition of faces and different methods of a resolution enhancement;
* Analysing the recognition error from method to method.


## data-acquisition-and-preparation

As a test case, I decided to choose the video as close to reality - taken directly from the [Ring.com device](https://ring.com/videodoorbells). After watching about [30 videos](https://www.youtube.com/channel/UCSDG3M0e2mGX9_qtHEtzj2Q/videos), I decided to choose the [one](https://www.youtube.com/watch?v=zwUeS_sXJcY) with the appearance and disappearance of several different people on the scene and long enough for generating data set. (I choose only one due low hardware resource and limited time conditions)

<p align="center">
  <img src="https://github.com/denis-r4/resolution-and-face-recognition/blob/master/media/data.png">
</p>

Next, using the OpenCV detector, I detect faces ROI on the video and removed undetected frames. 
As a model for face recognition, I chose the one based on VGG16, which has an input frame size 224x244. Detected ROI's were less than the required input size. Also, for more variability, I generated new noisy variants from each detected ROI , where the noise means downsampling, which is equivalent to different stance and remoteness of faces from a camera. 

224x224   |   200x200   |   180x180   |   150x150   |   120x120   |   100x100    
-------   |   -------   |   -------   |   -------   |   -------   |   -------
   <p align="center"> <img src="https://github.com/denis-r4/resolution-and-face-recognition/blob/master/media/resize_examples/224.png"> </p>   |   <p align="center"> <img src="https://github.com/denis-r4/resolution-and-face-recognition/blob/master/media/resize_examples/200.png"> </p>   |   <p align="center"> <img src="https://github.com/denis-r4/resolution-and-face-recognition/blob/master/media/resize_examples/180.png"> </p>   |   <p align="center"> <img src="https://github.com/denis-r4/resolution-and-face-recognition/blob/master/media/resize_examples/150.png"> </p>   |   <p align="center"> <img src="https://github.com/denis-r4/resolution-and-face-recognition/blob/master/media/resize_examples/120.png"> </p>   |   <p align="center"> <img src="https://github.com/denis-r4/resolution-and-face-recognition/blob/master/media/resize_examples/100.png"> </p>   

|   80x80   |   50x50   |   30x30   |   20x20   |   15x15   |   10x10   |   5x5   |
|   ------   |:------:|:------:|:------:|:------:|:------:|:------:|
|   <p align="center"> <img src="https://github.com/denis-r4/resolution-and-face-recognition/blob/master/media/resize_examples/80.png"> </p>   |   <p align="center"> <img src="https://github.com/denis-r4/resolution-and-face-recognition/blob/master/media/resize_examples/50.png"> </p>   |   <p align="center"> <img src="https://github.com/denis-r4/resolution-and-face-recognition/blob/master/media/resize_examples/30.png"> </p>   |   <p align="center"> <img src="https://github.com/denis-r4/resolution-and-face-recognition/blob/master/media/resize_examples/20.png"> </p>   |   <p align="center"> <img src="https://github.com/denis-r4/resolution-and-face-recognition/blob/master/media/resize_examples/15.png"> </p>   |   <p align="center"> <img src="https://github.com/denis-r4/resolution-and-face-recognition/blob/master/media/resize_examples/10.png"> </p>   |   <p align="center"> <img src="https://github.com/denis-r4/resolution-and-face-recognition/blob/master/media/resize_examples/5.png"> </p>   |


## face-recognition-model
For choosing face-recognition model I looked up a few free pre-trained models and stopped at the [Caffe model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo#vgg-face-cnn-descriptor) - [VGG16 for face recognition](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/). I prefer Theano, so, I repacked "caffemodel" to "pickle" format.
Then, I used fc7 layer instead of fc8 for getting vector representation of an input.


## tests
### Algorithms
For testing, I choose three most spread algorithms - [bilinear interpolation](https://en.wikipedia.org/wiki/Bilinear_interpolation), [nearest neighbor](https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation), and [bicubic interpolation](https://en.wikipedia.org/wiki/Bicubic_interpolation). Also after some searching of recent implementation of super-resolution (SR) algorithms based on neural networks, I found the good one - [neural-enhance](https://github.com/alexjc/neural-enhance)  
Also, there is a good SR benchmark that I found later - [sr-bechmark](https://github.com/huangzehao/Super-Resolution.Benckmark), but I didn't have a possibility to test more than one SR solution, so, I choose the first one.   

### Tests
As soon as I got everything all necessary parts, the tests started.
Firstly, I computed a vector representation of all generated samples in different resolution conditions.
Secondly, all received vectors have been passed through cosine distance (to vectors of images in database). 
Thirdly, I [got](https://github.com/denis-r4/resolution-and-face-recognition/blob/master/notebooks/accuracy.ipynb) an overall mean accuracy and a mean accuracy per each scale of source images. 
Resulting plot looks like this: 
<br/>
<p align="center">
  <img src="https://github.com/denis-r4/resolution-and-face-recognition/blob/master/media/accuracy_res.png">
</p>
The bottom axis represents a different scale of source images.
The left axis represents a received accuracy of face-recognition for particular scale.

## results-analysis
Considering the received results, it can be concluded that in the case of using particular [super-resolution model](https://github.com/alexjc/neural-enhance) and [face-recognition system](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/), it didn't give an accuracy benefit as compared with the interpolation methods, that have close to each other by accuracy and significantly higher than tested SR-method. However, the applying of artificial image resolution methods can increase the face-recognition accuracy, and let to skip less frames on a data-cleaning stage. 

Also, I want to point that given result has to be considered only for described above models. And doesn't have to block the interest to possibilities and accuracy benifits, which might bring the super-resolution technologies and related.  Another one thing, that I want to point is that SR-methods already successfully have been applied for detection and recognition accuracy growth in my past experience. Those SR methods have been based on the work - [Super-resolution from a Single Image]( http://www.wisdom.weizmann.ac.il/~vision/single_image_SR/files/single_image_SR.pdf)

## timing
In total, I spent on this research around 11 hours and up to 16 hours for computing hashes, rendering SR up-scaled frames on my laptop CPU. 
