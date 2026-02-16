# Genetic Programming-Based Evolutionary Deep

# Learning for Data-Efficient Image Classification

## Ying Bi,Member, IEEE, Bing Xue,Senior Member, IEEE, and Mengjie Zhang,Fellow, IEEE,

#### Abstract—Data-efficient image classification is a challenging

#### task that aims to solve image classification using small training

#### data. Neural network-based deep learning methods are effective

#### for image classification, but they typically require large-scale

#### training data and have major limitations such as requiring

#### expertise to design network architectures and having poor inter-

#### pretability. Evolutionary deep learning is a recent hot topic that

#### combines evolutionary computation with deep learning. However,

#### most evolutionary deep learning methods focus on evolving archi-

#### tectures of neural networks, which still suffers from limitations

#### such as poor interpretability. To address this, this paper proposes

#### a new genetic programming-based evolutionary deep learning

#### approach to data-efficient image classification. The new approach

#### can automatically evolve variable-length models using many

#### important operators from both image and classification domains.

#### It can learn different types of image features from colour or

#### gray-scale images, and construct effective and diverse ensembles

#### for image classification. A flexible multi-layer representation

#### enables the new approach to automatically construct shallow

#### or deep models/trees for different tasks and perform effective

#### transformations on the input data via multiple internal nodes.

#### The new approach is applied to solve five image classification

#### tasks with different training set sizes. The results show that it

#### achieves better performance in most cases than deep learning

#### methods for data-efficient image classification. A deep analysis

#### shows that the new approach has good convergence and evolves

#### models with high interpretability, different lengths/sizes/shapes,

#### and good transferability.

#### Index Terms—Evolutionary Deep Learning; Genetic Program-

#### ming; Image Classification; Small Data; Evolutionary Computa-

#### tion; Deep Learning

### I. INTRODUCTION

### Image classification tasks have a wide range of applications

### such as detecting cancer from x-ray images, identifying a

### face from a set of photographs, and classifying fish species

### from underwater images [1] [2] [3] [4]. Although image

### classification has been investigated for decades, it remains a

```
Manuscript received XXX; revised XXX; revised XXX; accepted XXX.
This work was supported in part by the Marsden Fund of New Zealand
Government under Contracts VUW1913, VUW1914 and VUW2115, the
Science for Technological Innovation Challenge (SfTI) fund under grant
E3603/2903, MBIE Data Science SSIF Fund under the contract RTVU1914,
and National Natural Science Foundation of China (NSFC) under Grant
```
61876169. (Corresponding author: Ying Bi)
    The author Ying Bi is with School of Electrical and Information En-
gineering, Zhengzhou University, Zhengzhou 450001, China, and School
of Engineering and Computer Science, Victoria University of Wellington,
Wellington 6140, New Zealand (e-mail: ying.bi@ecs.vuw.ac.nz).
    The authors Bing Xue and Mengjie Zhang are with School of
Engineering and Computer Science, Victoria University of Welling-
ton, Wellington 6140, New Zealand (e-mail: bing.xue@ecs.vuw.ac.nz;
mengjie.zhang@ecs.vuw.ac.nz).
    Color versions of one or more of the figures in this paper are available
online at [http://ieeexplore.ieee.org.](http://ieeexplore.ieee.org.)

### challenging task due to many factors, e.g., high inter-class

### similarity and intra-class dissimilarity, image distortion, and

### lack of sufficient training data. In recent years, data-efficient

### image classification that aims to use small data to perform

### effective image classification arise much attention. It can re-

### duce the reliance on a large number of training data, therefore

### reducing the cost and effort for data collection, labelling,

### prepossessing, and storage. Data-efficient image classification

### is also important for many applications in medicine, remote

### sensing and biology, where the labelled data are not easy to

### obtain due to the cost, privacy and security.

### Deep learning is a hot research area with many successful

### applications [5]. The goal of deep learning is to automatically

### learn or discover multiple levels of abstraction from data (i.e.

### representation learning) that are effective for a particular task

### [6]. These methods often include three main characteristics,

### i.e., sufficient model complexity, layer-by-layer data process-

### ing, and feature transformation [7]. Deep learning methods

### include both neural-network(NN)-based methods such as con-

### volutional NNs (CNNs) [8] and non-NN-based methods such

### as deep forest [7] and PCANet [9]. In recent years, deep

### CNNs become a dominant approach to image classification

### [8]. However, these NN-based methods have major limitations,

### such as requiring expensive computing devices to run, a large

### number of training data to train, and rich expertise to design

### the architectures [7]. Another important disadvantage of deep

### NNs is poor interpretability due to the large number of pa-

### rameters and very deep structures in the model [7]. To address

### these limitations, it is worth inventing new non-NN-based deep

### learning methods, which can not only maintain the diversity

### of the artificial intelligence/machine learning/computational

### intelligence research community but also bring new ideas to

### this community.

### Evolutionary deep learning (EDL) is a new research field

### that aims to use evolutionary computation (EC) techniques to

### evolve or optimise deep models [10] [11]. The advantages,

### i.e., population-based beam search, non-differential objective

### functions, ease of cooperating with domain knowledge, ro-

### bustness to dynamic changes, etc, have enabled EC methods

### to solve many complex optimisation and learning problems in

### various fields including finance, engineering, security, health-

### care, manufacturing, and business [11] [12].

### Existing works of EDL can be broadly classified into two

### categories, i.e., the use of EC methods to automatically to

### optimise deep NNs by searching for architectures, weights,

### loss functions etc, and the use of EC methods to automat-

### ically evolve variable-length, relatively deep, non-NN-based

### models. In the first category, many EC methods such as

## arXiv:2209.13233v1 [cs.NE] 27 Sep 2022


### genetic algorithms (GAs), genetic programming (GP), particle

### swarm optimisation (PSO), and evolutionary multi-objective

### optimisation (EMO) have been used to automatically evolve

### different types of deep NNs, e.g., CNNs, Autoencoders, re-

### current neural networks (RNNs), and generative adversarial

### networks (GANs) [10] [13] [14]. However, they often suffer

### from the limitations of deep NNs, such as requiring a large

### number of training examples, having “black-box” models with

### millions of parameters, and are computationally expensive. In

### the second category, to the best of our knowledge, the main

### method is GP, which can automatically evolve deep models

### to solve a problem [11]. According to the definition of deep

### learning [5] [7], GP is a deep learning method because it

### can automatically learn models with advanced operators that

### are sufficiently complex to perform data/feature transformation

### to solve tasks. There is also an increasing trend in recent

### works recognising GP as a deep learning method [10] [15].

### Unlike NN-based deep learning methods, GP-based EDL

### methods can automatically evolve variable-length models and

### their co-efficiencies without making any assumptions about

### the model structure/architecture [16]. Importantly, the models

### learned by GP often have varying complexity and potentially

### high interpretability, which are difficult to achieve in NNs.

### Benefiting from these advantages, GP-based EDL methods

### have been successfully applied to image classification with

### good performance [4] [17] [18]. However, the potential of GP

### has not been fully investigated. For tasks with small training

### data, limited computation resources and high requirements

### for model interpretability, where deep NNs have difficulty

### to address, it is necessary to investigate novel non-NN-based

### deep learning methods i.e. GP-based methods.

### Existing GP-based EDL methods have limitations in solving

### image classification, which can be broadly summarised into

### three aspects. First, most existing methods have only been

### examined on datasets with gray-scale images. Real-world

### problems may have images with various numbers of channels

### such as RGB channels, so it is necessary to develop a new

### GP method applicable to these different images. Second, most

### of the existing GP-based methods have rarely been applied

### to large-scale image classification datasets such as SVHN,

### CIFAR10, CIFAR100, and ImageNet. Third, the potential of

### GP with different operators to automatically learn models

### has not been fully investigated. The flexible representation

### allows GP to automatically build models using different types

### of simple and complex operators from different domains as

### functions (internal nodes) [19]. But there is still a lot of

### research potential in developing new representations including

### GP’s tree structures, the function and terminal sets.

### Therefore, this paper focuses on developing a new GP-based

### EDL approach to data-efficient image classification, where

### current NN methods can not provide satisfactory performance

### due to the requirement of large training data. To achieve

### this, we will develop a new GP-based EDL approach (i.e.

### EDLGP) with a new model/solution representation, a new

### function set and a new terminal set, enabling it to automatically

### learn features from gray-scale and/or colour images, select

### classification algorithms, and build effective ensembles for

### image classification. We will examine the performance of the

### new approach on well-known image classification datasets

### including CIFAR10, FashionMNIST, and SVHN and two

### face image datasets under the scenario of small training data.

### EDLGP will be compared with a large number of existing

### competitive methods including CNNs of varying architectures

### and complexity. To highlight the potential of GP-based EDL

### methods, we perform a comprehensive comparison between

### EDLGP and CNNs for image classification. In addition, a deep

### analysis on the results in terms of convergence behaviour,

### tree/model size, model interpretability and transferability is

### conducted.

### The main characteristics of the new EDLGP approach are

### summarised as follows.

### 1) The EDLGP approach can evolve variable-length tree-

### based symbolic models, achieving promising classifica-

### tion performance in the data-efficient scenario, which

### are difficult for current popular NN-based methods to

### address due to the requirement of a large number of

### training images. The design of automatically evolving

### ensembles further enhance the generalisation, which is

### a key issue when the training set is small.

### 2) Compared with existing methods, the EDLGP approach

### has a flexible multi-layer model representation to au-

### tomatically evolve shallow or deep models for differ-

### ent image classification tasks. With this representation,

### EDLGP can deal with gray-scale and/or colour images,

### extract image features via different processes such as

### image filtering and complex feature extraction, perform

### classification by using cascading and ensembles with

### high diversity, and automatically select parameters for

### image operators and classification algorithms. Unlike

### NN-based methods, EDLGP evolves models with small

### numbers of parameters, which are easy to learn and less

### relying on the scale of the data.

### 3) The EDLGP approach can evolve models with high

### interpretability and transferability. This is important for

### providing insights into the problems being solved and

### also for future model reuse and development.

### II. RELATEDWORK

### This section reviews related works on evolutionary deep

### learning including automatically evolving CNNs and GP for

### image classification and related works on data-efficient image

### classification. The limitations of these works are summarised.

### A. Evolutionary Deep Learning (EDL)

### 1) NN-based EDL methods:The methods that fall into this

### category typically use EC techniques to optimise deep NNs

### by searching for architectures, weights, and other parameters.

### A detailed review of existing work can be found in [10]

### [13] [14]. Sun et al. [20] proposed an evolutionary algorithm

### to search for CNN architectures and connection weights for

### image classification. In [21], a GA-based method was pro-

### posed to automatically design block-based CNN architectures

### for image classification. Lu et al. [22] developed an EMO-

### based method (i.e. NSGA-II) to automatically search CNN

### architectures while optimising multiple objectives. Zhu and Jin


### [23] proposed an EC-based neural architecture search (NAS)

### method under the real-time federated learning framework.

### The computational costs are significantly reduced by using

### this method. Zhang et al. [24] developed a method based on

### sampled training and node inheritance to improve the compu-

### tational efficiency of EC-based NAS for image classification.

### 2) GP-based EDL methods: Unlike NNs, GP can use

### a flexible tree-like symbolic representation to automatically

### evolve trees/models with different lengths to meet the needs

### of problem solving. For many problems, including image clas-

### sification, GP can automatically discover the representation

### of the data by evolving models of appropriate complexity.

### According to the definition of deep learning [5], GP can be

### a deep learning method because it can learn very complex

### models and transform the input data by using many internal

### nodes in the GP trees into higher-level data.

### An early review on GP-based EDL methods is presented

### in [11], which mentions typical GP methods such as 3-

### tier GP and multi-layer GP for image classification. These

### methods can automatically evolve models, extract features

### from raw images and perform classification, which is similar

### to CNNs. Rodriguez-Coayahuitl et al. [25] developed a GP

### autoencoder with structured layers to achieve representation

### learning and introduced the concept of deep GP. Shao et al.

### [26] proposed a GP-based method that can use different filters

### and image operators to achieve feature learning for image

### classification. This method achieved better performance than

### CNNs on several datasets. Bi et al. presented a series of GP-

### based methods, such as FLGP [27], FGP [19], and IEGP [28],

### for representation learning and image classification, where the

### evolved GP trees are constructed via multiple functional layers.

### B. Data-Efficient Image Classification

### Data-efficient image classification focuses on solving image

### classification with a small number of available training data,

### which has become an increasingly important topic in recent

### years. Bruintjes et al. [29] and Lengyel et al. [30] proposed

### the first and second well-known challenges on data-efficient

### deep learning, which aims to develop new deep learning

### methods using small training data to solve computer vision

### tasks including image classification, object detection, instance

### segmentation, and action recognition. In [29] [30], two im-

### portant rules of solving data-efficient image classification are

### proposed, i.e., 1) models can only be trained from scratch

### using the training set of the task, 2) other data, transfer learning

### and model pre-training are prohibited.

### Arora et al. [31] proposed a convolutional neural tangent

### kernel (CNTK) method, e.g., classifying CIFAR10 with 10

### to 640 training images. The results showed that the CNTK

### methods with different numbers of layers can beat ResNet.

### Brigato and Iocchi [32] investigated different CNNs of vary-

### ing complexity for image classification on small data. The

### results showed that dropout can improve the generalisation of

### CNNs. Bi et al. [33] proposed a multi-objective GP method

### to simultaneously optimise training accuracy and a distance

### measure to improve generalisation of the classification system

### using a small training set.

### Barz and Denzler [34] proposed the well-known Cosine loss

### for training deep NNs using small datasets. By maximising

### the cosine similarity between the outputs of the NNs and the

### one-hot vectors of the true class, this loss function gained

### better results than the commonly used cross-entropy loss on

### several image datasets. Brigato et al. [35] proposed eight

### data-efficient image classification benchmarks including object

### classification, digit recognition, medical image classification,

### and satellite image classification. Several methods were inves-

### tigated to solve these benchmarks. The results showed param-

### eter tuning in terms of batch size, learning rate and weight

### decay can improve the performance of existing methods on

### these datasets. Sun et al. [36] proposed visual inductive priors

### framework with a new neural network architecture for data-

### efficient image classification. A loss function based on the

### positive class classification loss and the intra-class compact-

### ness loss was developed to further improve the generalisation.

### This method ranked the first in the first data-efficient image

### classification challenge [29]. Zhao and Wen [37] proposed a

### method with a two-stage manner to train a teacher network

### and a student network for data-efficient image classification.

### This method ranked the second in the challenge [29].

### C. Summary

### Existing work shows the potential of GP for data-efficient

### image classification [17] [38] [39]. However, all these methods

### focus on relatively simple tasks and use gray-scale images.

### Furthermore, very few GP-based methods have been tested on

### commonly used datasets, such as CIFAR10. The comparison

### between GP-based EDL methods and NN-based deep learning

### methods is not sufficient and comprehensive to show the

### advantages of GP in solving data-efficient image classification.

### To address these limitations and further explore the potential

### of GP-based EDL methods, this paper proposes a new data-

### efficient image classification approach based on GP and con-

### ducts comprehensive comparisons between the new approach

### and CNN to demonstrate its effectiveness.

### III. THEPROPOSEDAPPROACH

### This section introduce the proposed EDLGP approach to

### image classification, including the overall algorithm, the indi-

### vidual representation, genetic operators, and fitness function.

### To highlight the advantages of the new approach, a detailed

### comparison between it with deep CNNs is presented.

### A. Overall Algorithm

### An image classification task often consists of a training

### set and a test set. The training set is denoted asDtrain=

### {(xi,yi)}mi=1and the test set is denoted asDtest={xj}nj=1,

### wherex∈RG×W×Hdenotes the images with a size ofW×H

### and a channel ofG, andy∈Zdenotes the class label of

### the image (the total number of classes isC). The number

### of training images ismand the number of testing images is

### n. The task is to useDtrainto build a modelg(x)that can

### effectively predictyfor testing/unseen imagesDtest.

### The overall process of EDLGP for image classification is

### shown in Fig. 1. The input of the system is the training set


#### Training set

#### Genetic Programming learning

#### Best GP tree

```
Population
initialisation
```
```
Fitness
evaluation Terminate?
```
```
Elitism Crossover Mutation
```
```
swap replace
```
#### Test set Classification

```
New representation
```
```
Population
generation
No
```
```
Yes
Start End
```
#### Accuracy of

#### Ensemble the test set

```
Selection
```
```
airplanebird ship cat
```
Fig. 1. The overall algorithm of the proposed EDLGP approach to image classification.

### Dtrainand the output is the best GP tree/modelg(x). EDLGP

### starts by randomly initialising a number of tree-based models

### according to the predefined representation, the function set

### and the terminal set. The population is evaluated using the

### fitness function, where each model/tree/individual is assigned

### with a fitness value i.e. the accuracy of the training set.

### During the evolutionary process, promising individuals are

### selected using a selection method and new individuals are

### generated using genetic operators i.e. subtree crossover and

### subtree mutation. A new population is created by copying a

### proportion of individuals with the highest fitness values and

### by generating new individuals from crossover and mutation

### operations. The process of population generation, fitness eval-

### uation and selection is repeated until reaching a predefined

### termination criterion. After the evolutionary process, the best

### GP tree/model is returned. The best treeg∗(x)is used to

### classify images in the test set.

### The optimisation process of the EDLGP approach can be

### defined as follows

### g∗(x) = argmax

```
f∈F, t∈T
```
### L(g(x),Dtrain) (1)

### where g(x)represents a GP tree/model and L represents

### the objective/fitness function.Fdenotes the functions andT

### denotes the terminals, which are employed to construct GP

### models/trees. The best model is learned by using EDLGP with

### a goal of maximisingLusing the training setDtrain. Note that

### Lcan be a loss function as that in NNs to be minimised.

### To find the best modelg∗(x), a new representation, a new

### function set, and a new terminal set are developed in EDLGP.

### These new components allow EDLGP to automatically evolve

### variable-length models that extract informative image features

### and build effective ensembles of classifiers for classification.

### B. Components of EDLGP

### The EDLGP approach has a new model representa-

### tion/encoding, making it different from existing GP methods

### [4] [11]. Furthermore, EDLGP uses genetic operators to create

### new populations of trees and a fitness function for evaluating

### the new models. These components are introduced as follows.

### 1) New Model Representation: EDLGP uses a tree-based

### structure based on strongly typed GP [40] to represent the

### model for image classification. The tree structure and two

### example trees are shown in Fig. 2. Specifically, the tree

### structure consists of nine concept layers, i.e., input, image

### filtering, feature extraction, concatenation, classification and

### cascade, concatenation, classification, summation, and output.

### The input layer represents the inputs of a GP tree, such as raw

### images and the parameters of operators. The image filtering

### layer uses a set of commonly used image filters to process

### the input image. The feature extraction layer contains well-

### known image descriptors that can generate a set of features

### from images. The concatenation layers aim to concatenate

### the features generated from the previous layer to create a

### more comprehensive feature set. The classification and cascade

### layers perform classification and cascade the predicted class

### labels or weights to the input feature vector to form a stronger

### feature set. This idea was borrowed from cascade learning

### [41]. The features generated by the classification and cascade

### layer can be further concatenated by a concatenation layer. The

### classification layer performs classification on the image using

### the features from its child nodes. The summation layer adds

### the predicted “probabilities” from all its child nodes/classifiers

### for each class. The output layer performs the majority voting

### to make a prediction on which class the image belongs to, i.e.

### the class label.

### The tree structure of EDLGP consists of flexible layers and

### fixed layers with a good balance between the necessary model

### complexity of dealing with an image classification task and

### the flexibility of the model growing in depth. Specifically, the

### input, feature extraction, classification, summation, and output

### layers are fixed, which follows a commonly used manner for

### building an ensemble for image classification. These layers

### must appear in each GP tree/model, but the functions used

### in those layers vary with trees/models. The flexible layers

### are the image filtering layer, the concatenation layer, and


```
Summation
```
```
Classification
```
```
Input
```
```
Feature extraction
```
```
Image filtering LoG
```
```
HOG
```
```
Comb
```
```
Hist
```
```
Sum
```
```
Gray Blue
```
```
Concatenation
```
```
Concatenation
```
```
Classification and cascade
```
```
CC_SVM
```
```
Comb
```
```
Sobel
```
```
Sobel_FE
```
```
Comb
```
```
LBP_FE
```
```
Green Gray
```
```
CC_LR
```
```
RF
```
```
Hist
```
```
Mean Red
```
```
Max
```
```
Gau
2
```
```
Concept
layers
```
```
A shallow model A deep model
```
```
Mean
```
```
HOG_FE
```
```
Comb
```
```
SIFT
```
```
Sum
```
```
Red Green
```
```
Majority voting
```
```
SVM
```
```
LBP
```
```
Gray
```
```
ERF
```
```
Sobel
```
```
100
```
```
30
1
```
```
Max
```
```
Variable
depth
```
```
Tree structure
```
```
LoG
```
```
SIFT
```
```
Red
```
```
CC_SVM
```
```
CC_SVM
```
```
100
```
```
30
```
```
ERF
```
```
Variable
depth
```
```
Majority voting
```
```
Output
Flexible layer
```
```
Fixed layer
```
Fig. 2. The tree structure of EDLGP and two example trees/models. The tree structure consists of a number of concept layers, where each layer uses a set
of predefined functions that can be searched by EDLGP to build the model. The tree structure allows EDLGP to construct shallow and deep models (as the
two examples) with different functions, sizes and shapes to solve different tasks. Each tree can extract features from raw images and construct ensembles of
classifiers to achieve effective classification.

### the classification and cascade layer, which are optional for

### constructing GP trees/models. These flexible layers allow

### EDLGP to generate more effective features by using different

### numbers of functions as internal nodes in the GP trees. This

### is important for dealing with difficult image classification

### tasks that require sufficiently complex data transformation to

### generate informative features.

### There are four main characteristics of the new model

### representation. First, it allows EDLGP to evolve shallow and

### deep models for different tasks. A simple model may only

### have a few internal and leaf nodes, similar to the example

### tree shown in the middle part of Fig. 2. A deep model may

### be constructed by a large number of internal nodes, as the

### example tree is shown in the right part of Fig. 2. Second,

### this representation allows EDLGP to construct ensembles of

### classifiers/ensembles for image classification, which enhances

### the effectiveness of the models when the number of training

### images is small. Third, this representation allows the con-

### structed model to automatically learn effective image features

### with different processes, i.e., image filtering, feature extraction

### based on commonly used image descriptors, and cascading

### after classification, which are common in computer vision and

### machine learning. Fourth, EDLGP can evolve ensembles with

### high diversity for image classification. The ensembles include

### cascade classifiers and classifiers built using features extracted

### by the corresponding subtree. The diversity ensures the effec-

### tiveness of the constructed ensembles. These characteristics

### make EDLGP significantly different from existing GP-based

### methods for image classification.

### 2) Functions and Terminals: The functions represent the

### internal and root nodes and the terminals represent the leaf

### nodes of GP trees/models. Different from existing GP meth-

### ods, EDLGP uses more functions to extract different types of

### features and new classification and cascade functions to evolve

### deep models according to the model representation.

### Allterminalsare listed in Table I. The terminals include the

### input image to be classified and parameters for corresponding

### functions. If the input images are colour, the terminals will be

##### TABLE I

##### TERMINALS

```
Terminal Description
Red,Blue,
Green
```
```
The red, blue, and green channels of the colour image. Each
of them is a 2D array with values in range [0, 1]
gray The gray-scale image, which is a 2D array with the values
in range [0, 1]
t The number of trees in RF, ERF, CCRF, and CCERF. It
is an integer in range [50, 1000] with a step of 50
d The maximal tree depth in RF, ERF, CCRF, and CCERF.
It is an integer in range [10, 100] with a step of 10
f The frequency of Gabor and GaborFE. Its value is in range
[π/ 8 ,π/2]with a step ofπ/ 2
```
##### √

##### 2

```
θ The orientation of Gabor. Its value is in range[0,^78 π]with
a step ofπ 8
o 1 ,o 2 The derivative orders of GauD and GauDFE. It is an integer
in range[0,2]
σ The standard deviation of Gau and GauFE. It is an integer
in range[1,3]
```
### Red, Blue, Green, and gray, representing three colour channels

### and the gray-scale channel. If the input images are gray-scale,

### the terminal will be gray. The remaining terminals aret,d,

### f,θ,o 1 ,o 2 , andσ, which have their value ranges according

### to commonly used settings. Their values are automatically

### selected during the evolutionary process.

### Theimage filteringlayer includes commonly used image

### filtering operators, i.e., Mean, Median, Min, Max, Gau, GauD,

### Lap, LoG1, LoG2, Sobel, and Gabor. Functions at this layer

### also include the operators that can process an image, such as

### HOGF, LBPF, Sqrt, and ReLU. In addition, the functions

### include AddMaxP and SubMaxP, which take two images

### as inputs, perform 2 × 2 max-pooling to the images, add

### and subtract the two small images to generate a new image,

### respectively. If the size of the two input images is not the same,

### the two functions will perform max-pooling only on the small

### image to make the size the same before sum or subtraction.

### This is possible since the size of images can only be reduced

### by using the 2×2 max-pooling operation in the function set.

### By using these different operators, it is expected to generate

### more effective features from images. Table II lists all functions

### and the process is illustrated in Fig. 3.

### Thefeature extraction layer contains 11 functions listed


##### TABLE II

##### IMAGE FILTERING FUNCTIONS

```
Function Description
Mean Perform 3×3 mean filtering
Median Perform 3×3 median filtering
Min Perform 3×3 minimum filtering
Max Perform 3×3 maximum filtering
Gau Perform Gaussian filtering and the standard deviation isσ
GauD Generate a new image by calculating derivatives of Gaussian
filter with standard deviationσand orderso 1 ando 2
Lap Perform Laplacian filtering
LoG1 Perform Laplacian of Gaussian filtering, and the standard
deviation is 1
LoG2 Perform Laplacian of Gaussian filtering, and the standard
deviation is 2
Sobel Perform 3×3 Sobel filtering to the input image
Gabor Perform Gabor filtering, where the orientation isθand the
frequency isf
LBPF Generate a LBP image
HOGF Generate an HOG image
Sqrt Return sqrt root value of each value of the input image. Return
1 if the value is negative
ReLU Perform the operation using rectified linear unit on the input
image
AddMaxP Perform 2 × 2 max-pooling on two input images or the smaller
image and add the two images
SubMaxP Perform 2 × 2 max-pooling on two input images or the smaller
image and subtract the two images
```
```
An image
```
```
An image
filtering function,
e.g., Gau, GauD
```
```
Gau
```
```
GauD
```
```
Gabor
```
```
Sobel
```
```
LBP_F
```
```
HOG_F
```
```
LoG
```
```
LoG
```
```
Lap
```
Fig. 3. Illustration of image filtering and the images obtained by applying
some functions.

### in Table III that can extract different types of features from

### images. The process is illustrated in Fig. 4. These functions

### include commonly used image descriptors, i.e., Histogram

### (Hist), HOG, LBP, HOGFE, LBPFE, and GaborFE, and

### methods for detecting edges, i.e., SobelFE and GauDFE,

### and others, i.e., Conca and GauFE. The Hist, HOG and LBP

### functions extract features from the input image. The XFE

### functions perform corresponding filtering operations on the

### image and extract the features by concatenating the image into

### a vector. The Conca function can concatenate two images to

### form a feature vector.

##### TABLE III

##### FEATURE EXTRACTION FUNCTIONS

```
Function Description
Conca Concatenate two images into a feature vector
Hist Extract 256 histogram features
HOG Extract HOG features from the input image [42]. The fea-
tures are the mean values of all 4 × 4 grids from the image
LBP Extract uniform LBP features [43]. The radius is 1.5 and the
number of neighbours is 8 in LBP.
SIFT Extract 128 dense SIFT features from the input image [44]
LBPFE,
HOGFE
```
```
Generate a LBP or HOG image and concatenate the image
into a feature vector
SobelFE,
GaborFE,
GauFE,
GauDFE
```
```
Use the Sobel, Gabor, Gau, or GauD filter to process the
input image and concatenate the new image into a feature
vector
```
### Thefeature concatenationlayer uses Comb2, Comb3 and

### Comb4 to further concatenate the feature vectors from their 2,

### 3, and 4 child nodes into a feature vector, respectively, which

```
A
```
```
n^
```
```
im
```
```
ag
```
```
e
```
```
A set of features
```
```
A feature extraction function
e.g., HOG, LBP, SIFT
```
```
Fig. 4. Illustration of feature extraction.
```
```
A feature vector
```
```
A feature vector
```
```
A feature vector
```
```
Feature concatenation
e.g.,Comb
```
```
Fig. 5. Illustration of feature concatenation using function Comb2.
```
### increases the comprehension of the features for describing an

### image. This layer can increase the width of the GP trees and

### produce more features for the classification layer. The process

### is illustrated in Fig. 5.

### The classification and cascadelayer has four functions

### listed in Table IV, i.e., cascade random forest (CCRF),

### cascade extremely randomised trees (CCERF), cascade logis-

### tic regression (CCLR), and cascade support vector machine

### (CCSVM), which perform classification and concatenate the

### predicted class probabilities with the input features to form the

### output features, as shown in Fig. 6. This follows the concept of

### cascade ensemble learning [7]. The prediction of an instance is

### aC-dimensional vector denoting the probabilities for aC-class

### problem. If the number of features isfn, the number of output

### features from these functions will befn+C. Unlike CCRF,

### CCERF and CCLR, which are soft classifiers, CCSVM

### is a hard classifier so that the prediction vector is binary

### (discrete). To build effective classifiers, the main parameters

### of CCRF and CCERF, i.e., the number of trees and the

### maximal tree depth, are set as terminals and their values can

### be automatically selected by EDLGP.

```
A^
```
```
fe
```
```
at
```
```
ur
```
```
e^
```
```
ve
```
```
cto
```
```
r
```
```
A trained
classifier
```
```
A^
```
```
ne
```
```
w^
```
```
fe
```
```
at
```
```
ur
```
```
e^
```
```
ve
```
```
cto
```
```
r
```
```
Predict class
probabilities Cascade
```
```
Classification and cascade
```
```
C class
```
```
Fig. 6. Illustration of classification and cascade.
```
### Theclassificationlayer contains functions, i.e., RF, ERF,

### LR, and SVM, which take features as inputs and return the

### predicted class probabilities. Thesummationlayer has three

### functions, i.e., Sum2, Sum3 and Sum4, which take two, three

### and four vectors of predicted probabilities as inputs and add

### them, respectively. Each vector denotes the probabilities of all

### classes an instance belongs to. These functions are listed in

### Table IV. Theoutputlayer can make predictions according to

### the output of GP trees by assigning the class label with the

### highest probability to the input image.

### Besides the above functions/operators, it is possible to use

### other different functions at the corresponding layer in EDLGP.

### In other words, EDLGP can be easily extended by using more

### functions. The representation and the search mechanism allow

### EDLGP to automatically select the functions and terminals to

### build trees/models. However, it is suggested to carefully set the

### function set to make a good balance between the effectiveness

### of the potential models and the search space.


##### TABLE IV

##### CLASSIFICATION AND CASCADE,CLASSIFICATION AND SUMMATION

```
FUNCTIONS
```
```
Function Description
CCRF,
CCERF,
CCLR,
CCSVM
```
```
Use the RF, ERF, LR, and SVM methods to perform classifi-
cation. The outputs of these functions are the concatenation of
the input features and the predicted class probabilities
```
```
RF Random forest classification method. The number of trees ist
and the maximal tree depth isd
ERF Extremely randomised trees classification method. The number
of trees istand the maximal tree depth isd
LR Logistic regression classification method
SVM Support vector machine classification method
Sum2/3/4 Sum 2/3/4 vectors of predicted probabilities
```
### 3) Genetic Operators: During the evolutionary process,

### new GP trees are generated using genetic operators, i.e.,

### subtree crossover and subtree mutation. The subtree crossover

### operator is conducted on two selected trees/parents. It ran-

### domly selects two subtrees from the parents and swaps the two

### subtrees to generate two new trees, as shown in Fig. 7. Note

### that the two selected subtrees must have the same output types

### in order to generate two feasible trees. The subtree mutation

### operator is conducted on one selected tree/parent. It randomly

### selects a subtree of the parent and replaces the subtree with a

### randomly generated subtree, as shown in Fig. 8.

```
Parent 1 Parent 2 Offspring 1 Offspring 2
```
```
Crossover
Swap
```
Fig. 7. Illustration of crossover operation.

```
A randomly Parent 1
generated subtree
```
```
Offspring 1
```
```
Replace Mutation
```
Fig. 8. Illustration of mutation operation.

### 4) Fitness Evaluation: A fitness function is used in the

### fitness evaluation process to evaluate the goodness of the GP

### trees/models to the tasks. For a classification task, the most

### commonly used fitness function is the classification accuracy

### to be maximised, which is defined as

### L=

### Ncorrect

### Ntotal

### ×100% (2)

### where Ncorrect denotes the number of correctly classified

### instances andNtotaldenotes the total number of instances in

### the training set. The fitness function is calculated using k-fold

### cross-validation on the training set. In EDLGP, the value ofK

### is set tomin(3,nc), wherencdenotes the number of instances

### in the smallest class of the training set. Ifnc= 1, the fitness

### function will be the training accuracy. Note that this fitness

### function is effective for balanced classification. For unbalanced

### classification, other measures such as balanced accuracy can

### be used as the fitness function for EDLGP.

### C. Comparisons Between EDLGP and CNN

### The proposed EDLGP approach can automatically learn

### features and evolve ensembles for image classification. To

### highlight the characteristics of EDLGP, this section provides

### detailed analysis and comparisons between EDLGP (as a

### typical example of GP-based EDL methods) and CNNs, which

### is a dominant deep learning method for image classification.

### The mainsimilaritiesare summarised as follows.

### 1) CNNs and EDLGP can automatically learn features and

### perform classification from raw images.

### 2) CNNs and EDLGP are learning algorithms that learn

### models of many layers, which can perform complex data

### transformation to achieve good performance.

### 3) CNNs and EDLGP use image operations such as con-

### volution and pooling to learn features from raw images.

### The maindifferencesare summarised as follows.

### 1) EDLGP uses a variable-length tree-based structure to

### represent a model, while CNNs typically uses a fixed-

### length layer-wise structure to represent a model. The

### flexible representation allows EDLGP to automatically

### learn models with different shapes, sizes, and depths. In

### contrast, CNNs need to predefine the model structure

### and length before training.

### 2) EDLGP is open to incorporate domain knowledge in a

### way of adding functions/operators to the corresponding

### layers, which cannot be easily achieved in CNNs.

### 3) EDLGP has fewer model parameters and running param-

### eters than CNNs. Detailed comparisons in terms of the

### running parameters are listed in Table V. The number

### of CNN model parameters depends on the architecture

### configuration, which is often much larger than that of

### the automatically evolved EDLGP models.

### 4) The models evolved by EDLGP have potentially higher

### interpretability than CNN models. The EDLGP model

### is composed of functions/operators from the image and

### machine learning domains, which can provide domain

### knowledge. The CNN model is composed of layers of

### a huge number of parameters, which are not straightfor-

### wardly to explain.

### 5) The feature extraction and classification processes of

### CNNs and EDLGP are different. CNNs usually extract

### and construct features via a number of convolutional

### and pooling layers with learned kernels, and perform

### classification using softmax. EDLGP extracts features

### using image operators with predefined kernels and uses

### a traditional classification algorithm to perform clas-

### sification. This makes CNNs and EDLGP applicable

### and effective for image classification under different

### scenarios. CNNs requires sufficient training data to learn

### effective feature maps and sub-sampling maps in these

### predefined layers to achieve promising performance, i.e.,

### suitable for large-scale image classification. The EDLGP

### approach is more effective than CNNs when the training

### data is small, i.e., data-efficient image classification.

### 6) The CNN methods can greatly benefit from running on

### GPU, while EDLGP would not at the current stage.

### IV. EXPERIMENTDESIGN

### This section describes the experiment design, including

### datasets, the comparison methods, and parameter settings.


##### TABLE V

##### COMPARISONS OF RUNNING PARAMETERS OFEDLGPANDCNNS

```
Convolutional neural networks Evolutionary deep learning based on
genetic programming
Type of activation functions: Type of selection:
ReLU, tanh, sigmoid, etc Tournament selection or Roulette
wheel selection
Architecture configurations: Architecture configurations:
No. hidden layers Function set
No. convolutional layers Terminal set
No. pooling layers Optimisation configurations:
No. other layers No. generations
How to connect these layers Population size
No. features maps/nodes Probability of elitism/reproduction
Kernel sizes Probability of mutation
Optimisation configurations: Probability of crossover
Learning rate minimal and maximal tree size
Dropout: 0.25/0.50 selection size
Momentum
L1/L2 weight regularisation
penalty
```
```
Tree generation method: full, grow,
ramped-half-and-half
Weight initialisation: uniform,
Xavier Glorot, etc.
Batch size: 32/64/
```
### A. Image Classification Datasets

### To evaluate the effectiveness of the new approach, five

### different datasets are used, which are CIFAR10 [45], Fash-

### ionMNIST (FMNIST in short) [46], SVHN [47], ORL [48],

### and Extended Yale B [49]. The CIFAR10, FMNIST, and

### SVHN datasets are object classification tasks. The ORL and

### Extended Yale B datasets are face classification tasks. The

### example images of these datasets are shown in Figs. 9 and 10.

### The CIFAR10 dataset is composed of 50,000 training im-

### ages and 10,000 testing images. The images are colour and of

### size 32×32. The FashionMNIST dataset has 60,000 training

### images and 10,000 testing images in gray-scale. The image

### size is 28×28. The SVHN dataset contains 73,257 colour

### images for training and 26,032 colour images for testing.

### The image size is 32×32. The ORL dataset consists of 400

### facial images from 40 people, i.e. 10 images per person. The

### Extended Yale B dataset has 2,424 facial images in 38 classes.

##### CIFAR

```
Fashion_MNIST
```
##### SVHN

Fig. 9. Example images of the CIFAR10, FashionMNIST, and SVHN
datasets. Each image represents one class.

### The EDLGP approach is examined on data-efficient image

### classification problems. A small number of training images are

### used in the experiments. For the CIFAR10, FashionMNIST

### and SVHN datasets, 10, 20, 40, 80, 160, 320, 640, and

### 1280 images are randomly selected from each class in the

### original training set to form a small training set used in

### the experiments, following the settings in [32]. The small

### datasets are called sCIFAR10-X, sFMNIST-X and sSVHN-

```
ORL
```
```
Extended Yale B
```
```
Fig. 10. Example images of the ORL and Extended Yale B datasets.
TABLE VI
SUMMARY OF THE DATASETS
```
```
Dataset #Class Image
size
```
```
Train set size Test set size
```
##### CIFAR10 10 32 × 32 × 3 50,000 10,

```
FashionMNIST 10 28 × 28 60,000 10,
SVHN 10 32 × 32 × 3 73,257 26,
ORL 40 56 × 46 400
Extended Yale B 38 32 × 32 2424
```
### X, where X denotes the number of training images in each

### class. To comprehensively show the performance, we also test

### EDLGP using very small training sets of CIFAR10, i.e., 1,

### 2, 4, 8, 16, 32, 64, and 128 images per class, following the

### settings in [31]. In all these settings,the original test sets

### are used for testing. For the ORL dataset, 2, 3, 4, and 5

### images are randomly selected to form the training sets and the

### remaining images are used for testing in the experiments [50],

### respectively. For the Extended Yale B dataset, 15, 20, 25, and

### 30 images are randomly selected for training and the rest are

### used for testing [41], respectively. As a result, a large number

### of experiments are conducted to comprehensively demonstrate

### the performance of EDLGP in comparison with different deep

### learning methods.

### B. Comparison Methods

### We compare EDLGP with existing methods including CNNs

### with different architectures, common methods, and non-NN-

### based deep learning methods, which are very effective methods

### on the corresponding datasets.

- On sCIFAR10, sFMNIST and sSVHN, the comparisons

### methods are CNN-lc [32], CNN-mc [32], CNN-hc [32],

### and ResNet-20 [51]. CNN-lc, CNN-mc and CNN-hc

### denote different CNN methods with low, middle and

### high complexity used for image classification. Different

### Dropout rates are investigated in these methods and the

### details can be found in [32].

- On few-shot sCIFAR10, the comparison methods are

### ResNet [51], 5-layer convolutional neural tangent kernel

### (CNTK) networks [31], 8-layer CNTK [31], 11-layer

### CNTK [31], 14-layer CNTK [31], and Harmonic Net-

### works [52].

- On ORL, the comparisons methods are Eigenface based

### on PCA [53], Fisherface based on LDA [54], Laplacian-

### face [55], neighbourhood preserving embedding (NPE)

### [56], marginal Fisher analysis (MFA) [57], CNN [50],

### and deep metric learning based on CNN and k nearest

### neighbour classification (KCNN) [50].

- On Extended Yale B, the comparisons methods are col-

### laborative representation classifier (CRC) [58], SRC [59],


### correntropy-based sparse representation (CESR) [60], ro-

### bust sparse coding (RSC) [61], half-quadratic with the

### additive form (HQA) [62], half-quadratic with the mul-

### tiplicative form (HQM) [62], NMR [63], robust matrix

### regression (RMR) [64], Fisher discrimination dictionary

### learning (FDDL) [65], low-rank shared dictionary learn-

### ing (LRSDL) [66], DCM based on NMR (DCM(N)) [41],

### and DCM based on SRC (DCM(S)) [41].

### The parameter settings of these comparison methods refer to

### the corresponding references. The other experimental settings

### refer to [41] [32] [50]. The goal of this paper is to explore the

### potential of GP-based EDL for data-efficient image classifica-

### tion. The aforementioned methods are used for comparisons

### because they are effective methods on these datasets using

### small training sets. We also compare the EDLGP approach

### with more state-of-the-art data-efficient deep learning methods

### mentioned in [35] on CIFAR10 under different settings. Due

### to the page limit, the results and analysis is presented in

### the supplementary materials. We do not include GP-based

### methods for comparisons because they have not been applied

### or not designed for solving these datasets.

### C. Parameter Settings

### The parameter settings for the EDLGP approach follow

### the commonly used settings in the GP community [4] [67].

### The maximal number of generations is set to 50 and the

### population size is set to 100. The crossover rate is 0.5, the

### mutation rate is 0.49, and the elitism rate is 0.01. The selection

### method is tournament selection with size 5. The population

### initialisation method is the ramped-half-and-half method. The

### initial tree depth is 2-10. The maximal tree depth is 10. Note

### that these parameter values for EDLGP can be fine-tuned on

### each dataset for optimal settings. However, our study aims to

### investigate a more general approach that can achieve good

### performance using a common setting. In the experiments,

### EDLGP is executed 10 times independently on each dataset

### using different random seeds because of the stochastic nature

### and the high computation cost.

### V. RESULTS ANDDISCUSSIONS

### This section compares the performance of the EDLGP

### approach with the benchmark methods on different datasets of

### various numbers of training images. It also deeply analyses dif-

### ferent aspects of EDLGP in terms of convergence behaviours,

### interpretability and transferability of models/trees.

### A. Classification Accuracy

### Overall Performance: In total, there are 40

### cases/experiments (i.e., 3×8 on sCIFAR10, sFMNIST

### and sSVHN, 8 on few-shot sCIFAR10, and 2×4 on ORL

### and Extended Yale B) to compare the performance EDLGP

### with different CNNs and existing methods. All the test

### results are listed in Tables VII - X. In all 40 cases, EDLGP

### achieves better accuracy in 30 cases among all the existing

### methods, indicating that EDLGP is effective on these

### different image classification tasks by automatically evolving

### variable-length/depth models. Note that the compared

### methods are the state-of-the-art methods and their results

### are from the corresponding references. Compared with

### these methods, EDLGP achieves better performance on the

### few-shot sCIFAR10, ORL and Extended Yale B datasets in

### all scenarios with small numbers of training images. On

### sCIFAR10, sFMNIST and sSVHN, EDLGP achieves better

### performance when the training set is small, and slightly worse

### performance when the size of the training set is large. The

### results show that EDLGP is more effective for data-efficient

### image classification, particularly the training set is very small.

### Results on sCIFAR10, sFMNIST and sSVHN:The clas-

### sification accuracies on the test sets are listed in Table VII.

### Note that the test sets are the original one of CIFAR10,

### FashionMNIST and SVHN and the results of these compar-

### ison methods are from [32]. On sCIFAR10, EDLGP achieves

### the best accuracy among all 11 methods including ResNet-

### 20 in 6 cases out of 8 cases, i.e., worse on sCIFAR

### with 1280 training images per class. On very small training

### data scenarios, EDLGP achieves maximal accuracy that is

### much higher than the best one of all comparison methods on

### sCIFAR10. For example, 3.9% higher (35.8% vs. 31.9%) on

### sCIFAR10-10, 6.2% higher (43.2% vs. 37.0%) on sCIFAR10-

### 20, 5.5% higher (48.1% vs. 42.6%) on sCIFAR10-40, 4.2%

### higher (52.3% vs. 48.1%) on sCIFAR10-80, and 4.3% higher

### (58.2% vs. 53.9%) on sCIFAR10-160. The results further

### demonstrate that EDLGP is very effective when the training set

### is small. On FashionMNIST, EDLGP achieves the best results

### on sFMNIST-20, sFMNIST-40, sFMNIST-80, and sFMNIST-

### 640. In the remaining cases, EDLGP achieves slightly worse

### accuracy than the best accuracy, i.e., 0.1% gap on sFMNIST-

### 10, 1.5% gap on sFMNIST-160, 0.6% gap on sFMNIST-

### 640, and 1.5% gap on sFMNIST-1280. The results show

### that EDLGP is effective for classifying the sFMNIST dataset.

### On sSVHN, EDLGP achieves the best results on sSVHN-10,

### sSVHN-20, sSVHN-40, and sSVHN-80. More importantly, the

### best accuracy obtained by EDLGP is much higher than that

### by all comparison methods, i.e., 27.9% higher on sSVHN-

### 10, 19.5% higher on sSVHN-20, and 8.1% on sSVHN-40.

### With the increasing of training data, some CNNs and ResNet-

### 20 methods tend to achieve better classification accuracy and

### the accuracy gap between EDLGP and the best CNN be-

### come smaller. On sSVHN-160, sSVHN-320, sSVHN-640, and

### sSVHN-1280, EDLGP achieves better results than the majority

### of the comparison methods and worse results than ResNet-20.

### The results show that CNNs require sufficient training data to

### obtain good classification performance. Compared with these

### CNN methods, EDLGP is significantly less data intensive

### and can achieve better performance when the training set is

### very small. CNN models typically contain a huge number

### of parameters so that a large number of training images

### are needed to train. Unlike CNNs, EDLGP automatically

### evolves tree-based models of functions and operators that have

### significantly fewer parameters and does not require a large

### number of training images. EDLGP is more data efficient

### than CNNs since it can achieve better performance than CNNs

### when the training data is small.

### Results on few-shot sCIFAR10: Table VIII shows the


##### TABLE VII

##### MEANCLASSIFICATION ACCURACY(%)OFCNNS OF VARYING COMPLEXITY AND CLASSIFICATION ACCURACY(%)OFEDLGPON SMALLCIFAR10,

##### FASHIONMNISTANDSVHNDATASETS OF VARYING TRAINING SET SIZE

```
Model Dropout sCIFAR10-10 sCIFAR10-20 sCIFAR10-40 sCIFAR10-80 sCIFAR10-160 sCIFAR10-320 sCIFAR10-640 sCIFAR10-
CNN-lc 0.0 27.1 32.4 36.1 41.8 45.3 50.1 54.1 59.
CNN-lc 0.4 29.7 33.8 38.2 43.5 47.1 52.8 57.2 61.
CNN-lc 0.7 29.7 34.9 40.0 44.9 49.4 53.9 58.1 62.
CNN-mc 0.0 28.5 34.3 38.8 43.0 48.6 53.7 58.8 63.
CNN-mc 0.4 29.7 34.9 39.9 45.4 50.9 55.8 61.0 66.
CNN-mc 0.7 31.5 36.2 41.3 47.1 51.9 57.7 62.8 67.
CNN-hc 0.0 30.1 34.2 39.1 44.7 50.6 56.1 61.2 65.
CNN-hc 0.4 31.7 36.1 40.8 46.5 52.0 58.0 63.5 68.
CNN-hc 0.7 31.9 37.0 42.5 48.1 53.9 59.5 64.8 69.
ResNet-20 – 23.3 29.0 31.9 38.5 44.7 51.3 62.3 71.
EDLGP(best) – 35.8 43.2 48.1 52.3 58.2 60.6 64.8 64.
EDLGP(mean) – 31.7±3.8 37.7±4.0 45.7±1.6 49.3±1.8 54.3±2.5 58.7±1.5 61.6±2.0 61.1±3.
Model Dropout sFMNIST-10 sFMNIST-20 sFMNIST-40 sFMNIST-80 sFMNIST-160 sFMNIST-320 sFMNIST-640 sFMNIST-
CNN-lc 0.0 71.1 74.0 77.8 81.0 83.6 85.4 86.7 88.
CNN-lc 0.4 71.2 76.1 79.8 81.9 84.2 85.7 87.1 88.
CNN-lc 0.7 71.3 75.6 78.7 81.6 83.3 85.3 87.0 87.
CNN-mc 0.0 72.5 75.5 79.0 82.0 84.4 86.3 88.0 89.
CNN-mc 0.4 72.4 76.1 79.6 82.9 84.7 86.6 88.1 89.
CNN-mc 0.7 72.5 76.9 79.9 82.9 84.9 86.8 88.2 89.
CNN-hc 0.0 71.9 75.9 80.1 82.3 85.1 86.8 88.6 89.
CNN-hc 0.4 72.2 76.3 80.2 83.0 85.0 86.9 88.5 89.
CNN-hc 0.7 73.3 77.4 80.5 83.2 86.5 87.1 88.8 89.
ResNet-20 – 62.3 71.4 77.0 80.4 84.1 86.9 89.2 90.
EDLGP(best) – 73.2 79.5 82.1 83.5 85.0 87.2 88.6 89.
EDLGP(mean) – 71.8±1.0 78.0±0.7 81.2±0.7 83.0±0.4 84.5±0.5 86.1±0.7 86.8±0.6 88.0±0.
Model Dropout sSVHN-10 sSVHN-20 sSVHN-40 sSVHN-80 sSVHN-160 sSVHN-320 sSVHN-640 sSVHN-
CNN-lc 0.0 25.3 37.5 50.5 64.1 73.0 77.7 80.9 83.
CNN-lc 0.4 28.4 44.9 59.0 70.2 77.0 80.3 83.3 85.
CNN-lc 0.7 28.8 46.7 60.6 72.1 77.7 81.3 83.2 86.
CNN-mc 0.0 26.8 39.9 53.4 68.6 75.5 79.6 83.2 85.
CNN-mc 0.4 29.6 43.3 64.3 72.1 78.3 82.2 84.8 87.
CNN-mc 0.7 27.7 45.8 64.1 74.6 79.7 83.2 86.2 88.
CNN-hc 0.0 24.9 37.5 55.5 67.7 74.5 80.1 84.0 86.
CNN-hc 0.4 27.5 45.6 63.1 73.6 79.3 82.7 85.7 88.
CNN-hc 0.7 28.8 44.8 64.7 74.4 79.5 84.0 86.3 88.
ResNet-20 – 20.3 40.0 54.7 74.1 83.5 86.7 89.5 92.
EDLGP(best) – 57.5 66.2 72.8 75.5 79.7 82.1 82.1 85.
EDLGP(mean) – 55.5±1.5 60.2±2.4 70.1±1.5 73.6±1.1 76.9±2.2 79.0±1.5 80.5±0.8 82.3±0.
```
##### TABLE VIII

##### CLASSIFICATION ACCURACY(%)OFEDLGPAND BENCHMARK METHODS OF VARYING COMPLEXITY ON FEW-SHOTCIFAR

```
Model sCIFAR10-1 sCIFAR10-2 sCIFAR10-4 sCIFAR10-8 sCIFAR10-16 sCIFAR10-32 sCIFAR10-64 sCIFAR10-
ResNet [31] 14.59 17.50 19.52 23.32 28.30 33.15 41.66 49.
5-layer CNTK [31] 15.08 18.03 20.83 24.82 29.63 35.26 41.24 47.
8-layer CNTK [31] 15.24 18.50 21.07 25.18 30.17 36.05 42.10 48.
11-layer CNTK [31] 15.31 18.69 21.23 25.40 30.46 36.44 42.44 48.
14-layer CNTK [31] 15.33 18.79 21.34 25.48 30.48 36.57 42.63 48.
Harmonic Networks [52] 11.87 22.24 26.28 34.94 40.47 49.59 56.69 63.
EDLGP(best) 21.10 23.63 27.42 29.93 41.50 45.25 50.53 56.
EDLGP(mean) 15.22±2.71 17.69±3.97 20.63±4.86 25.51±3.57 37.91±1.89 41.11±3.88 47.21±2.98 54.04±1.
```
### results on few-shot sCIFAR10 with 1, 2, 4, 8, 16, 32, 64, 128

### training images per class, respectively. Among all the methods

### including ResNet, EDLGP achieves the best classification

### accuracy on few-shot sCIFAR10 with 1-128 training images

### per class. The best accuracy obtained by EDLGP is much

### higher than the best accuracy of all the comparison methods,

### i.e., over 7% on average. In these few-shot settings, both the

### performance of CNN and EDLGP increase with the number

### of training images, but EDLGP performs better than these

### CNN methods. The results continually show the effectiveness

### of EDLGP in data-efficient image classification.

### Results on ORL and Extended Yale B:On the face

### datasets, EDLGP achieves higher maximum and average ac-

### curacies than any of the comparison methods in all scenarios.

### Specifically, EDLGP improves the average accuracy by 8.8%

##### TABLE IX

##### CLASSIFICATION ACCURACY(%)OFEDLGPAND THE BENCHMARK

##### METHODS ON THEORLDATASET

```
Method ORL-2 ORL-3 ORL-4 ORL-
Eigenface [53] 70.7±2.7 78.9±2.3 84.2±2.1 87.9±2.
Fisherface [54] 75.5±3.3 86.1±1.9 91.6±1.9 94.3±1.
Laplacianface [55] 77.6±2.5 86.0±2.0 90.3±1.7 93.0±1.
NPE [56] 77.6±2.7 85.7±1.8 90.5±1.8 93.4±1.
MFA [57] 75.4±3.1 86.1±1.9 91.6±1.9 94.3±1.
CNN [50] 69.7±3.1 82.9±2.5 87.9±1.8 91.5±2.
KCNN [50] 72.8±3.1 84.6±2.6 91.7±2.4 94.6±1.
EDLGP(best) 90.3 97.9 99.6 99.
EDLGP(mean) 86.4±3.9 96.4±1.0 98.4±1.1 99.3±0.
```
### on ORL-2, 10.3% on ORL-3, 6.7% on ORL-4, and 4.7% on

### ORL-5. On ORL-5, EDLGP achieves an average accuracy

### of 99.3% and maximum accuracy of 99.5%. Furthermore,

### compared with these seven methods, EDLGP has a smaller


##### TABLE X

##### CLASSIFICATION ACCURACY(%)OFEDLGPAND THE BENCHMARK

##### METHODS ON THEEXTENDEDYALEBDATASET. 15, 20, 25,AND 30

```
DENOTE DIFFERENT NUMBERS OF TRAINING IMAGES IN EACH CLASS
Method 15 20 25 30
CRC [58] 91.39 94.26 95.91 97.
SRC [59] 91.72 93.71 95.56 96.
CESR [60] 77.92 83.42 85.68 88.
RSC [61] 95.01 97.04 97.81 98.
HQA [62] 93.39 93.99 90.19 92.
HQM [62] 91.14 94.15 95.29 96.
NMR [63] 93.50 96.29 97.57 98.
RMR [64] 93.56 94.08 92.15 92.
FDDL [65] 93.44 94.92 96.38 96.
LRSDL [66] 94.92 96.69 97.88 98.
DCM(N) [41] 93.17 95.97 97.38 98.
DCM(S) [41] 98.87 99.51 99.63 99.
EDLGP(best) 99.89 99.94 99.93 99.
EDLGP(mean) 99.80±0.05 99.80±0.09 99.83±0.08 99.82±0.
```
### standard deviation value, which means EDLGP is more stable.

### On Extended Yale B, EDLGP gains over 99% accuracy in

### all scenarios. Among all the comparison methods, EDLGP

### achieves the best accuracy on Extended Yale B, which means

### that EDLGP is more accurate than the 12 different comparison

### methods. To sum up, EDLGP is effective for face image

### classification using few training images.

### To summarise, EDLGP achieves promising performance in

### image classification with small training data. Unlike compar-

### ative methods, EDLGP can simultaneously and automatically

### search for the best feature extraction methods and ensemble

### models to achieve effective classification. The learned EDLGP

### models have fewer parameters than CNNs so that they can

### be better trained on small training data and achieve higher

### classification accuracy. A large number of comparisons show

### that EDLGP is significantly more data efficient than CNNs

### and other well-known methods on different types of image

### classification tasks. The results comprehensively verify the

### effectiveness and superiority of EDLGP in data-efficient image

### classification.

### B. Running and Classification Time

### We take the few-shot sCIFAR10-X (X ranges from 1 to

### 128) dataset as a typical example to analyse the running and

### classification time of EDLGP. The running time of EDLGP on

### the other datasets is analysed in the supplementary materials

### due to the page limit. Figure 11 shows an average running

### time and classification time of EDLGP on a single CPU.

### The running time of EDLGP is less than 10 hours when the

### number of training images per class is smaller than 16. The

### running time of EDLGP increases gradually with the number

### of training images. On sCIFAR10-128, it uses more than

### 90 hours to complete the evolutionary learning process. The

### running time of EDLGP is clearly longer than that of CNNs,

### although we did not directly compare them. The main reason

### is that EDLGP is a population-based search algorithm that is

### currently implemented using the DEAP package running on

### CPUs. In contrast, CNNs can benefit from a fast computing

### facility–GPU, and use a shorter time for model training. The

### running time of EDLGP can be accelerated by using a GPU

### implementation or running it in parallel on multiple CPUs.

```
Fig. 11. Running time and test/classification time of EDLGP on the sCIFAR
dataset using 1-128 images/instances per class for training.
```
```
Fig. 12. Convergence behaviours of EDLGP on ORL and Extended Yale B.
```
### The classification/test time of EDLGP on sCIFAR10-X

### is fast, i.e., less than 4 minutes in all scenarios. In the

### classification process, EDLGP trains the model found via

### evolution to classify 10,000 images. The GP model com-

### plexity is the main factor that affects the classification time.

### From Fig. 11, EDLGP may learn a more complex model on

### sCIFAR10-64 and sCIFAR10-128 than the other scenarios,

### as it uses a longer classification time. The analysis of the

### model length/complexity will be conducted in the following

### subsection. Overall, EDLGP needs a reasonably long running

### time to complete the evolutionary process but uses a short time

### for classification.

### C. Convergence Behaviours

### We take the ORL and Extended Yale B datasets as examples

### to show the convergence behaviour of EDLGP. Note that

### its convergence behaviour on other datasets shows similar

### patterns. Figure 12 shows that EDLGP has good search ability

### and can converge to a high fitness value (i.e. accuracy on the

### training set) after 50 generations. Using different training set

### sizes, EDLGP achieves different fitness values during evolu-

### tion. Specifically, more training data corresponds to higher

### fitness values of EDLGP on these two datasets. This is because

### the fitness function evaluates the generalisation of the learned

### model by using k-fold cross-validation on the training set. To

### sum up, EDLGP has good search ability and convergence, and

### can find the best model through evolution.

### D. Interpretability of Trees/Models

### This subsection analyses model length/complexity and vi-

### sualisation to demonstrate the high interpretability of models

### learned by EDLGP.

### Tree/Model Size: The average tree size of EDLGP on

### sCIFAR10, ORL and Extended Yale B is shown in Fig. 13.

### On sCIFAR10 with 1-128 training images, the average tree


```
sCIFAR
```
Fig. 13. The average size (number of nodes) of trees evolved by EDLGP on
sCIFAR10, ORL, and Extended Yale B of different training set sizes.

```
Min
```
```
SIFT LBP
```
```
Gray
```
```
Blue
```
```
ERF
```
```
CC_SVM
```
```
ERF
```
```
Sum
```
```
ERF ERF
```
```
CC_SVM
```
```
Gau_FE
```
```
Gray Sub_MaxP
```
```
Gray
```
```
SIFT
```
```
Gray
```
```
Comb
```
```
Conca
```
```
Gabor
```
```
Blue Blue
```
```
2
```
```
1
```
```
4
```
```
LBP
```
```
Gray Sqrt
```
```
Red
```
```
Comb
```
```
HOG_FE
```
```
Sqrt
```
```
SIFT
```
```
HOG_F
```
```
Green Blue
```
```
Comb
```
```
Sobel_FE
```
```
Gabor
```
```
3
```
```
4
```
```
500
```
```
20
500
```
```
40 85070
350
```
```
40
```
Fig. 14. An example tree of EDLGP on sCIFAR10 using 80 training images
per class. This tree achieves 52.26% accuracy on the test set, which is better
than any of the comparison methods.

### size ranges from 42.6 to 73.8. EDLGP tends to gradually

### increase its tree size with the number of training images on

### sCIFAR10, which is a difficult task. On the ORL and Extended

### Yale B datasets, the average tree size ranges from 35.5 to 62.

### In case 1 (2 training images on ORL and 15 training images

### on Extended Yale B), EDLGP has a small initial fitness value

### (as shown in Fig. 12) and seems to improve tree quality by

### gradually increasing the tree size. When using more training

### images on these two datasets (i.e. in cases 3 and 4), EDLGP

### reaches a higher fitness value in initial generations and finds

### relatively smaller trees than that in case 1. To sum up, EDLGP

### can evolve variable-length trees on different datasets with

### various numbers of training images.

### Tree/Model Visualisation:Figure 14 shows an example

### tree of EDLGP on sCIFAR10-80. This tree is an ensemble

### of four ERF ensemble classifiers. Each branch can build one

### ERF classifier using different features generated from the

### corresponding child nodes and different parameter settings

### (i.e. number of decision trees and maximal tree depth). It

### processes images using Min, SubMaxP, Gabor, Sqrt, and

### HOGF operators on the input image in the gray, red, blue,

### and green channels, and extracts features using SIFT, LBP,

### GauFE, Conca, HOGFE, and SobelFE operators. CIFAR

### is a complex object classification dataset so that different types

### of features are extracted to improve classification performance.

### By using such a model, EDLGP achieves 52.26% test ac-

### curacy, which is the best accuracy among all the methods

### on sCIFAR10 using only 80 training images per class. More

### importantly, the size of this tree is much smaller than those

### CNNs.

### To further demonstrate the interpretability of models/trees

### evolved by EDLGP, a small tree on sFMNIST-640 is analysed.

```
SIFT
```
```
CC_RF
```
```
LR
```
```
CC_RF
```
```
CC_RF
```
```
LR
```
```
Sum
```
```
Sobel_FE
```
```
Gray
```
```
800
```
```
60
```
```
500
```
```
60
```
```
500
```
```
60
```
```
Gray
```
```
SIFT
```
```
CC_RF
```
```
LR
```
```
Sum
```
```
Sobel_FE
```
```
CC_RF
```
```
CC_RF
```
```
LR
```
```
Sum
```
```
Fig. 15. An example tree evolved by EDLGP on sFMNIST-640 and
visualisation of the test data transformed by each internal node of the tree.
T-SNE [68] is used for visualisation. Each plot is drawn using 100 testing
images per class of sFMNIST and each colour represents one class.
```
### The small tree is shown in Fig. 15. It achieves 86.7% test

### accuracy on sFMNIST-640. We use T-SNE [68] to visualise

### the data (i.e., features) generated by each internal node of the

### GP tree. For better visualisation, 100 test images per class are

### used and each test image is fed into the GP tree. We collect

### all the outputs of each internal node by feeding these test

### images and use T-SNE to perform visualisation on the outputs,

### as shown in Fig. 15. This example tree has two branches to

### generate an ensemble of LR classifiers. For the left branch, it

### extracts SIFT features, uses CCRF to generate a new feature

### vector, and performs classification using LR. The right branch

### extracts features using SobelFE, generates new features using

### two CCRF functions, and performs classification using LR.

### The visualised data in Fig. 15 shows that the function at each

### node makes significant changes to the input images/features.

### The data generated by the high-level nodes are more clustered.

### At the output node, the data generated from the Sum2 node

### are clearly clustered, which can reveal why this model can

### achieve high classification accuracy.

### To sum up, EDLGP evolves trees with different lengths,

### shapes and depths on different datasets. A single GP tree is

### a model that can perform image feature extraction and classi-

### fication using ensembles of classifiers. The functions/internal

### nodes of GP trees can make important transformations on the

### data to generate better ones for achieving good performance.

### The GP trees are easy to be visualised and some insights can

### be gained from them.

### E. Transferability of Trees/Models

### This subsection analyses the transferability of trees/models

### found by EDLGP to show the possibility of model reuse.

### The best EDLGP models found from sCIFAR10 using 1-

### training images per class is transferred to solve sSVNH using


### 10-1280 training images per class. The average accuracy (%) is

### presented in Table XI. The evolved EDLGP models are also

### transferred between the ORL and Extended Yale B datasets

### of different training set sizes. The average accuracy (%) is

### presented in Table XII. Note that in the transferring setting

### the classifiers of the GP trees learned on the source training

### set are re-trained using the training set of the target task and

### the GP trees are used to classify the test set of the target task.

##### TABLE XI

##### AVERAGE CLASSIFICATION ACCURACY(%)ON SSVHNUSING THE

##### MODELS LEARNED BYEDLGPFROM SCIFAR10-1TO SCIFAR10-128.

```
sCIFAR10−→sSVHN
10 20 40 80 160 320 640 1280
sCIFAR10-1 28.2 32.3 37.0 42.3 47.4 52.0 56.0 59.
sCIFAR10-2 23.3 26.9 29.5 32.2 34.1 35.9 37.5 38.
sCIFAR10-4 34.8 41.6 46.7 51.6 55.8 59.1 61.5 62.
sCIFAR10-8 27.9 31.9 36.3 39.4 42.8 45.9 48.5 50.
sCIFAR10-16 40.6 47.5 53.0 58.9 63.0 67.0 70.0 71.
sCIFAR10-32 46.8 54.4 60.0 65.0 68.1 71.3 73.5 75.
sCIFAR10-64 52.0 59.4 63.8 68.4 71.5 74.6 76.5 78.
sCIFAR10-128 47.2 55.5 61.4 67.6 71.4 75.2 77.5 79.
ResNet-20(original) 20.3 40.0 54.7 74.1 83.5 86.7 89.5 92.
EDLGP(original) 55.5 60.2 70.1 73.6 76.9 79.0 80.5 82.
```
### Table XI lists the classification results on sSVHN (target

### dataset) by reusing the best trees/models evolved by EDLGP

### on few-shot sCIFAR10 (source dataset). The final two rows

### list the original results of ResNet-20 and EDLGP learning

### from the target training set of sSVHN for comparisons. The

### results show that the learned EDLGP models have very

### strong transferability since they achieve better classification

### performance than ResNet-20 on sSVHN-10, sSVHN-20 and

### sSVHN-40 and very close accuracy to EDLGP in all cases.

### Comparing with other CNN methods in Table VII, the reused

### models can also obtain higher accuracy in most cases. The

### comparison can better demonstrate good transferability of

### the EDLGP models. From Table VII, we can also find that

### the transferability of the EDLGP models is also affected by

### the number of training instances in the source training set.

### Specifically, a relatively larger source training set can lead to

### better transferability of the EDLGP models on sSVHN. This

### is reasonable because both CIFAR10 and SVHN are object

### classification tasks and more data can lead to learn more

### generalised models. However, EDLGP does not heavily rely

### on increasing the the training set to improve its performance

### and the model transferability, since the models evolved on

### sCIFAR10-64 achieve the best performance on sSVHN-10,

### sSVHN-20, sSVHN-40, and sSVHN-80.

### Table XII shows the results obtained by transferring EDLGP

### models across the ORL and Extended Yale B datasets. The

### final row lists the results obtained by EDLGP using the

### target training set for model learning. From ORL to Extended

### Yale B, the reused EDLGP models achieve worse accuracy

### when using 15 training images per class, but very high

### accuracy (i.e. 88.6%) when using 30 training images per

### class. From Extended Yale B to ORL, the reused GP models

### obtain better accuracy than some comparison methods on ORL

### with different numbers of training images, e.g., better than

### Eigenface on all cases, better than CNN on ORL-2, ORL-

### and ORL4. The results can show the high transferability of the

### learned GP models across different face image classification

### tasks. Table XII shows that using different training set sizes

### in the source and target tasks, the models achieve different

### performances. In the case of ORL−→Extended Yale B, models

### learned from a small training set of ORL can achieve better

### performance on Extended Yale B. In the case of Extended Yale

### B−→ORL, models learned from 20 source training images

### per class achieve better performance on ORL with different

### training set sizes. The two face image datasets are different

### in terms of image variations, which may affect the generality

### and transferability of the models learned by EDLGP.

### To sum up, the results show that the models/trees learned by

### EDLGP have high transferability. An important reason is that

### the learned models/trees consisting of operators and algorithms

### can represent not only domain-specific information but also

### general-domain information, which enable them to have strong

### transferability. This reveals the high possibility of reusing the

### learned EDLGP models for solving other similar or related

### tasks. In addition, the results show that the source training

### set sizes may affect the transferability of the EDLGP models,

### which provides insights on effective model reuse.

##### TABLE XII

##### AVERAGE TEST ACCURACY(%)BY TRANSFERRINGGPMODELS

##### BETWEEN THEORLANDEXTENDEDYALEBDATASETS USING

```
DIFFERENT NUMBERS OF TRAINING IMAGES
```
```
ORL−→Extended Yale B Extended Yale B−→ORL
15 20 25 30 2 3 4 5
ORL-2 76.7 81.8 85.9 88.6 Extended.-15 62.3 73.8 80.3 83.
ORL-3 70.5 76.7 80.9 85.6 Extended.-20 71.2 83.8 89.3 91.
ORL-4 68.7 74.0 77.9 81.5 Extended.-25 55.5 68.1 76.2 81.
ORL-5 69.4 76.2 80.3 84.2 Extended.-30 69.6 82.6 88.0 91.
EDLGP 99.80 99.8 99.8 99.8 EDLGP 86.4 96.4 98.4 99.
```
### F. Summary

### The following observations of EDLGP as a typical exam-

### ple of GP-based EDL method can be summarised from the

### experimental analysis.

- EDLGP can achieve better performance than the com-

### pared methods in most cases on CIFAR10, Fash-

### ionMNIST, SVHN, ORL, and Extended Yale B. When

### the training set is very small, EDLGP can achieve state-

### of-the-art performance. This shows that EDLGP is an

### effective approach to data-efficient image classification.

- EDLGP shows good convergence and can find trees with

### different shapes, sizes and depths on different detasets.

### Compared with CNN-based image classification methods,

### EDLGP does not require manually tune/design/determine

### the model architectures and/or coefficients/parameters.

- EDLGP can evolve small and easy-interpretable trees to

### achieve high accuracy. The evolved EDLGP trees are

### composed of image and classification domain operators,

### which are interpretable to provide more insights into the

### tasks. This is difficult to be achieved by using CNN

### methods with numerous parameters.

- The trees/models evolved by EDLGP show high transfer-

### ability, facilitating model reuse and development in the

### future.


### VI. CONCLUSIONS

### In this paper, a GP-based EDL method, EDLGP, was

### proposed for automatically evolving variable-length models

### for image classification on small training data. A new rep-

### resentation was developed that includes a multi-layer tree

### structure, a function set and a terminal set, enabling EDLGP to

### efficiently build models to perform feature extraction, concate-

### nation, classification and cascade, and ensemble construction,

### automatically and simultaneously. The EDLGP approach has

### shown great potential in solving data-efficient image classi-

### fication by achieving better performance than many effective

### methods. A detailed analysis showed that the models learned

### by EDLGP have good convergence, high interpretability, and

### good transferability. Compared with existing popular CNN-

### based image classification methods, the EDLGP approach as a

### GP-based EDL approach has a number of advantages, such as

### without requiring expensive GPU devices to run and rich do-

### main expertise to design/tune model architectures/coefficients,

### data-efficient, and evolving variable-length models with high

### interpretability and transferability.

### This paper is a starting point showing the superiority of

### GP in comparisons with CNN-based methods for data-efficient

### image classification. More effective GP-based EDL methods

### can be developed to dig out the potential of GP in the future.

### REFERENCES

[1] J. Schindelin and et al., “Fiji: an open-source platform for biological-
image analysis,”Nature methods, vol. 9, no. 7, pp. 676–682, 2012.
[2] C. A. Schneider, W. S. Rasband, and K. W. Eliceiri, “Nih image to
imagej: 25 years of image analysis,”Nature methods, vol. 9, no. 7, pp.
671–675, 2012.
[3] D. Lu and Q. Weng, “A survey of image classification methods and
techniques for improving classification performance,”Int. J. Remote
Sens., vol. 28, no. 5, pp. 823–870, 2007.
[4] Y. Bi, B. Xue, and M. Zhang,Genetic Programming for Image Classifi-
cation: An Automated Approach to Feature Learning. Springer, 2021.
[5] Y. LeCun, Y. Bengio, and G. Hinton, “Deep learning,”nature, vol. 521,
no. 7553, pp. 436–444, 2015.
[6] Y. Bengio, A. Courville, and P. Vincent, “Representation learning: A
review and new perspectives,”IEEE Trans. Pattern Anal. Mach. Intell.,
vol. 35, no. 8, pp. 1798–1828, 2013.
[7] Z.-H. Zhou and J. Feng, “Deep forest,”Natl. Sci. Rev., vol. 6, no. 1, pp.
74–86, Oct 2018.
[8] L. Alzubaidi, J. Zhang, A. J. Humaidi, A. Al-Dujaili, Y. Duan, O. Al-
Shamma, J. Santamar ́ıa, M. A. Fadhel, M. Al-Amidie, and L. Farhan,
“Review of deep learning: Concepts, cnn architectures, challenges,
applications, future directions,”J. Big Data, vol. 8, no. 1, pp. 1–74,
2021.
[9] T.-H. Chan, K. Jia, S. Gao, J. Lu, Z. Zeng, and Y. Ma, “Pcanet: A
simple deep learning baseline for image classification?”IEEE Trans.
Image Process., vol. 24, no. 12, pp. 5017–5032, 2015.
[10] Z.-H. Zhan, J.-Y. Li, and J. Zhang, “Evolutionary deep learning: A
survey,”Neurocomputing, 2022.
[11] H. Al-Sahaf, Y. Bi, Q. Chen, A. Lensen, Y. Mei, Y. Sun, B. Tran, B. Xue,
and M. Zhang, “A survey on evolutionary machine learning,”J. Roy. Soc.
New Zeal., vol. 49, no. 2, pp. 205–228, 2019.
[12] T. B ̈ack, D. B. Fogel, and Z. Michalewicz, “Handbook of evolutionary
computation,”Release, vol. 97, no. 1, p. B1, 1997.
[13] X. Zhou, A. Qin, M. Gong, and K. C. Tan, “A survey on evolutionary
construction of deep neural networks,”IEEE Trans. Evol. Comput., 2021.
[14] Y. Liu, Y. Sun, B. Xue, M. Zhang, G. G. Yen, and K. C. Tan, “A survey
on evolutionary neural architecture search,”IEEE Trans. Neural Netw.
Learn. Syst., 2021, DOI: 10.1109/TNNLS.2021.3100554.
[15] A. D. Martinez, J. Del Ser, E. Villar-Rodriguez, E. Osaba, J. Poyatos,
S. Tabik, D. Molina, and F. Herrera, “Lights and shadows in evolutionary
deep learning: Taxonomy, critical methodological analysis, cases of
study, learned lessons, recommendations and challenges,”Information
Fusion, vol. 67, pp. 161–194, 2021.

```
[16] W. B. Langdon and R. Poli,Foundations of GeneticProgramming.
Springer Science & Business Media, 2013.
[17] Y. Bi, B. Xue, and M. Zhang, “Dual-tree genetic programming for few-
shot image classification,”IEEE Trans. Evol. Comput., vol. 26, no. 3,
pp. 555–569, 2022.
[18] ——, “Multi-objective genetic programming for feature learning in face
recognition,”Appl. Soft Comput., vol. 103, p. 107152, 2021.
[19] ——, “Genetic programming with image-related operators and a flexible
program structure for feature learning in image classification,”IEEE
Trans. Evol. Comput., vol. 25, no. 1, pp. 87–101, 2021.
[20] Y. Sun, B. Xue, M. Zhang, and G. G. Yen, “Evolving deep convolutional
neural networks for image classification,”IEEE Trans. Evol. Comput.,
vol. 24, no. 2, pp. 1–14, 2019.
[21] Y. Sun, B. Xue, M. Zhang, and G. G. Yen, “Completely automated cnn
architecture design based on blocks,”IEEE Trans. Neural Netw. Learn.
Syst., vol. 31, no. 4, pp. 1242–1254, 2019.
[22] Z. Lu, I. Whalen, Y. Dhebar, K. Deb, E. D. Goodman, W. Banzhaf, and
V. N. Boddeti, “Multiobjective evolutionary design of deep convolutional
neural networks for image classification,”IEEE Trans. Evol. Comput.,
vol. 25, no. 2, pp. 277–291, 2020.
[23] H. Zhu and Y. Jin, “Real-time federated evolutionary neural architecture
search,”IEEE Trans. Evol. Comput., vol. 26, no. 2, pp. 364–378, 2022.
[24] H. Zhang, Y. Jin, R. Cheng, and K. Hao, “Efficient evolutionary search
of attention convolutional networks via sampled training and node
inheritance,”IEEE Trans. Evol. Comput., vol. 25, no. 2, pp. 371–385,
2020.
[25] L. Rodriguez-Coayahuitl, A. Morales-Reyes, and H. J. Escalante, “Struc-
turally layered representation learning: Towards deep learning through
genetic programming,” inProc. EuroGP. Springer, 2018, pp. 271–288.
[26] L. Shao, L. Liu, and X. Li, “Feature learning for image classification via
multiobjective genetic programming,”IEEE Trans. Neural Netw. Learn.
Syst., vol. 25, no. 7, pp. 1359–1371, 2014.
[27] Y. Bi, B. Xue, and M. Zhang, “An effective feature learning approach
using genetic programming with image descriptors for image classifica-
tion,”IEEE Comput. Intell. Mag., vol. 15, no. 2, pp. 65–77, 2020.
[28] ——, “Genetic programming with a new representation to automatically
learn features and evolve ensembles for image classification,”IEEE
Trans. Cybern., vol. 51, no. 4, pp. 1769–1783, 2021.
[29] R.-J. Bruintjes, A. Lengyel, M. B. Rios, O. S. Kayhan, and J. van
Gemert, “Vipriors 1: Visual inductive priors for data-efficient deep
learning challenges,”arXiv preprint arXiv:2103.03768, 2021.
[30] A. Lengyel, R.-J. Bruintjes, M. B. Rios, O. S. Kayhan, D. Zam-
brano, N. Tomen, and J. van Gemert, “Vipriors 2: Visual induc-
tive priors for data-efficient deep learning challenges,”arXiv preprint
arXiv:2201.08625, 2022.
[31] S. Arora, S. S. Du, Z. Li, R. Salakhutdinov, R. Wang, and D. Yu,
“Harnessing the power of infinitely wide deep nets on small-data tasks,”
inProc. ICLR, 2019.
[32] L. Brigato and L. Iocchi, “A close look at deep learning with small
data,” inProc. IEEE ICPR, 2021, pp. 2490–2497.
[33] Y. Bi, B. Xue, and M. Zhang, “Using a small number of training
instances in genetic programming for face image classification,”Inf.
Sci., 2022.
[34] B. Barz and J. Denzler, “Deep learning on small datasets without pre-
training using cosine loss,” inProc. IEEE/CVF WCACV, 2020, pp. 1371–
1380.
[35] L. Brigato, B. Barz, L. Iocchi, and J. Denzler, “Tune it or don’t use it:
Benchmarking data-efficient image classification,” inProc. IEEE ICCV,
2021, pp. 1071–1080.
[36] P. Sun, X. Jin, W. Su, Y. He, H. Xue, and Q. Lu, “A visual inductive
priors framework for data-efficient image classification,” inProc. ECCV.
Springer, 2020, pp. 511–520.
[37] B. Zhao and X. Wen, “Distilling visual priors from self-supervised
learning,” inProc. ECCV. Springer, 2020, pp. 422–429.
[38] H. Al-Sahaf, M. Zhang, and M. Johnston, “Binary image classification:
A genetic programming approach to the problem of limited training
instances,”Evol. Comput., vol. 24, no. 1, pp. 143–182, 2016.
[39] Y. Bi, B. Xue, and M. Zhang, “Learning and sharing: A multitask genetic
programming approach to image feature learning,”IEEE Trans. Evol.
Comput., vol. 26, no. 2, pp. 218–232, 2022.
[40] D. J. Montana, “Strongly typed genetic programming,”Evol. Comput.,
vol. 3, no. 2, pp. 199–230, 1995.
[41] L. Zhang, J. Liu, B. Zhang, D. Zhang, and C. Zhu, “Deep cascade
model-based face recognition: When deep-layered learning meets small
data,”IEEE Trans. Image Process., vol. 29, pp. 1016–1029, 2019.
[42] N. Dalal and B. Triggs, “Histograms of oriented gradients for human
detection,” inProc. IEEE CVPR, vol. 1, 2005, pp. 886–893.
```

[43] T. Ojala, M. Pietikainen, and T. Maenpaa, “Multiresolution gray-scale
and rotation invariant texture classification with local binary patterns,”
IEEE Trans. Pattern Anal. Mach. Intell., vol. 24, no. 7, pp. 971–987,
2002.
[44] A. Vedaldi and B. Fulkerson, “VLFeat: An open and portable library of
computer vision algorithms,” inProc. 18th ACM Int. Conf. Multimedia,
2010, pp. 1469–1472.
[45] A. Krizhevsky, G. Hintonet al., “Learning multiple layers of features
from tiny images,” 2009.
[46] H. Xiao, K. Rasul, and R. Vollgraf, “Fashion-mnist: a novel image
dataset for benchmarking machine learning algorithms,”arXiv preprint
arXiv:1708.07747, 2017.
[47] Y. Netzer, T. Wang, A. Coates, A. Bissacco, B. Wu, and A. Y. Ng,
“Reading digits in natural images with unsupervised feature learning,”
2011.
[48] F. S. Samaria and A. C. Harter, “Parameterisation of a stochastic model
for human face identification,” inProc. IEEE Workshop Appl. Comput.
Vis., 1994, pp. 138–142.
[49] K.-C. Lee, J. Ho, and D. J. Kriegman, “Acquiring linear subspaces for
face recognition under variable lighting,”IEEE Trans. Pattern Anal.
Mach. Intell., no. 5, pp. 684–698, May 2005.
[50] T. Liao, Z. Lei, T. Zhu, S. Zeng, Y. Li, and C. Yuan, “Deep metric
learning for k nearest neighbour classification,”IEEE Trans. Knowl.
Data Eng., 2021.
[51] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image
recognition,” inProc. IEEE CVPR, 2016, pp. 770–778.
[52] M. Ulicny, V. A. Krylov, and R. Dahyot, “Harmonic networks with
limited training samples,” inProc. EUSIPCO. IEEE, 2019, pp. 1–5.
[53] M. Turk and A. Pentland, “Eigenfaces for recognition,”J. Cogn.
Neurosci., vol. 3, no. 1, pp. 71–86, 1991.
[54] P. N. Belhumeur, J. P. Hespanha, and D. J. Kriegman, “Eigenfaces vs.
fisherfaces: Recognition using class specific linear projection,”IEEE
Trans. Pattern Anal. Mach. Intell., vol. 19, no. 7, pp. 711–720, 1997.
[55] X. He, S. Yan, Y. Hu, P. Niyogi, and H.-J. Zhang, “Face recognition
using laplacianfaces,”IEEE Trans. Pattern Anal. Mach. Intell., vol. 27,
no. 3, pp. 328–340, 2005.
[56] X. He, D. Cai, S. Yan, and H.-J. Zhang, “Neighborhood preserving
embedding,” inProc. IEEE ICCV, vol. 2. IEEE, 2005, pp. 1208–1213.
[57] S. Yan, D. Xu, B. Zhang, H.-J. Zhang, Q. Yang, and S. Lin, “Graph
embedding and extensions: A general framework for dimensionality
reduction,”IEEE Trans. Pattern Anal. Mach. Intell., vol. 29, no. 1, pp.
40–51, 2006.
[58] L. Zhang, M. Yang, and X. Feng, “Sparse representation or collaborative
representation: Which helps face recognition?” inProc. ICCV, 2011, pp.
471–478.
[59] J. Wright, A. Y. Yang, A. Ganesh, S. S. Sastry, and Y. Ma, “Robust face
recognition via sparse representation,”IEEE Trans. Pattern Anal. Mach.
Intell., vol. 31, no. 2, pp. 210–227, 2008.
[60] R. He, W.-S. Zheng, and B.-G. Hu, “Maximum correntropy criterion
for robust face recognition,”IEEE Trans. Pattern Anal. Mach. Intell.,
vol. 33, no. 8, pp. 1561–1576, 2010.
[61] M. Yang, L. Zhang, J. Yang, and D. Zhang, “Robust sparse coding for
face recognition,” inProc. IEEE CVPR, 2011, pp. 625–632.
[62] R. He, W.-S. Zheng, T. Tan, and Z. Sun, “Half-quadratic-based iterative
minimization for robust sparse representation,”IEEE Trans. Pattern
Anal. Mach. Intell., vol. 36, no. 2, pp. 261–275, 2013.
[63] J. Yang, L. Luo, J. Qian, Y. Tai, F. Zhang, and Y. Xu, “Nuclear
norm based matrix regression with applications to face recognition with
occlusion and illumination changes,”IEEE Trans. Pattern Anal. Mach.
Intell., vol. 39, no. 1, pp. 156–171, 2016.
[64] J. Xie, J. Yang, J. J. Qian, Y. Tai, and H. M. Zhang, “Robust nuclear
norm-based matrix regression with applications to robust face recogni-
tion,”IEEE Trans. Image Process., vol. 26, no. 5, pp. 2286–2295, 2017.
[65] M. Yang, L. Zhang, X. Feng, and D. Zhang, “Sparse representation
based fisher discrimination dictionary learning for image classification,”
Int. J. Comput. Vis., vol. 109, no. 3, pp. 209–232, 2014.
[66] T. H. Vu and V. Monga, “Fast low-rank shared dictionary learning for
image classification,”IEEE Trans. Image Process., vol. 26, no. 11, pp.
5160–5175, 2017.
[67] Y. Bi, B. Xue, and M. Zhang, “An evolutionary deep learning approach
using genetic programming with convolution operators for image clas-
sification,” inProc. IEEE CEC, 2019, pp. 3197–3204.
[68] L. v. d. Maaten and G. Hinton, “Visualizing data using t-sne,”J. Mach.
Learn. Res., vol. 9, pp. 2579–2605, 2008.


