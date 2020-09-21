# chestXray-PGGAN
High Resolution Chest X-ray Image Synthesis Using Progressive-Growing Generative Adversarial Networks

**This repository is under construction.**

## Description

*For a detailed description of the work, please refer to the following article:*


**Ganesan P, Rajaraman S, Long R, Ghoraani B, Antani S. Assessment of Data Augmentation Strategies Toward Performance Improvement of Abnormality Classification in Chest Radiographs. In 2019 41st Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC) 2019 Jul 23 (pp. 841-844). IEEE.**


*Note: If you found this repository helpful, please consider citing the above article.*


PG-GAN was introduced by Karras et al.,[1] as an attempt to synthesize high-resolution images (upto 1024x1024 pixels), which were not realizable by DC-GANs. The primary reason for the difficulty in achieving high-resolution images before the introduction of PG-GANs was that the weights could not be learned consistently while searching for the global minimum using a network of that size. To circumvent this limitation, the weights in PG-GAN are learned by progressively growing the network starting from a 4x4 up to the final resolution. For example, in case of an image with a resolution of 512x512 pixels, the image is first downsampled into resolutions of 4x4, 8x8, 16x16, and all the way to 512x512. Then, the network is trained starting from the smallest resolution. After training the smallest resolution, the network is grown to the next resolution and the training is continued using the previous weights. In other words, the weights of the smaller resolutions are fully learned before switching the network to the higher resolution, thus making it possible for the network to reach the global minimum for the higher resolution weights. The network model is shown in Figure 1. More details on the PG-GAN model can be found in Ref.[1]. We utilize this model to synthesize high-resolution CXR images as explained in the following section.

![Figure 1: Random Samples During Training](https://github.com/prash030/chestXray-PGGAN/blob/master/pggantraining.png)

The goal of training a PG-GAN in this work is to perform GAN-based Augmentation (GA) for an abnormality classifier, using realistic CXR images of normal and abnormal classes. The dataset used in this study is made available for the Radiological Society of North America (RSNA) machine learning challenge (https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data) by the joint effort of radiologists from the RSNA and Society of Thoracic Radiology (STR). The dataset includes images with pulmonary opacity that may represent pneumonia or other disorders and other images with no abnormal findings. All the images were of 1024x1024 pixel dimensions with an 8-bit depth. The images were first pre-processed by segmenting the lung region of interest (ROI) using a dropout UNET. This segmentation helps to remove irrelevant regions that carry structures that do not contribute to the abnormality, so that the PG-GAN can focus on learning the ROI. The UNET model consists of dropout layers following a Gaussian distribution, after every pair of convolution and ReLU layers. The addition of Gaussian noise is expected to mimic the noise present during CXR image acquisition. The resulting images were cropped to a bounding box containing the lungs and were then resized to 512x512 pixels. These 512x512 images (8,954 normals and 11,653 abnormals) were used for training the PG-GAN.

The normal and abnormal images were randomly selected from the RSNA collection, and split into equal number (N = 6268) for training set while the rest (N = 2686 for normals and N = 5385 for abnormals) were used for test set. The PG-GAN was individually trained on the normal and abnormal images in the training set. Each training phase took about six days on a high-performance machine with an NVIDIA GTX 1080Ti GPU and 48GB RAM.

**The model weights after the network was fully trained, which took about 150 epochs, is available in the following Google Drive location: https://drive.google.com/drive/folders/1QsOWk6xU9cV2Qeg6uRSVMZP_AVgro0d6?usp=sharing**

## Instructions to Use the Model

*Detailed instructions on training, testing and image synthesis can be found in Terro Karras' github:* https://github.com/tkarras/progressive_growing_of_gans

### Dataset Preparation:
dataset_tool.py [-h] create_from_images datasets/tfrec_rsnaNormal trainingset/rsnaNormal


The above command will grab the images from “trainingset/rsnaNormal” folder and convert them to TF records. The TF records will be saved in datasets/tfrec_rsnaNormal. 

### Training:
python train.py


Training parameters should be mentioned in config.py. 


Statement from Karras' Github: “By default, config.py is configured to train a 1024x1024 network for CelebA-HQ using a single-GPU. This is expected to take about two weeks even on the highest-end NVIDIA GPUs.”


Edit config.py for rsnaNormal 512x512:


* data_dir = 'datasets'
* result_dir = 'results' # Make a directory named “results” to store the results
* desc = 'pgan' # this is just a string to label what kind of dataset you run
* desc += '-rsnaNormal';   dataset = EasyDict(tfrecord_dir='rsnaNormal')
* desc += '-preset-v2-1gpu'; num_gpus = 1; sched.minibatch_base = 4; sched.minibatch_dict = {4: 128, 8: 128, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8, 512: 4}; sched.G_lrate_dict = {512: 0.0015}; sched.D_lrate_dict = EasyDict(sched.G_lrate_dict); train.total_kimg = 15000
* desc += '-fp32'; sched.max_minibatch_per_gpu = {256: 16, 512: 8}


From Karras' Github: “The training script saves a snapshot of randomly generated images at regular intervals in fakes*.png and reports the overall progress in log.txt.”

### Synthesizing Images:
python train.py


train = EasyDict(func='util_scripts.generate_fake_images', run_id='000-pgan-rsnaNormal-preset-v2-1gpu-fp32', num_pngs=6300); num_gpus = 1; desc = 'fake-images-' + str(train.run_id)


This synthesizes images from the network saved in “networks/” folder.

# References:
[1] Karras T, Aila T, Laine S, Lehtinen J. Progressive growing of gans for improved quality, stability, and variation. arXiv preprint arXiv:1710.10196. 2017 Oct 27.


[2] Ganesan P, Rajaraman S, Long R, Ghoraani B, Antani S. Assessment of Data Augmentation Strategies Toward Performance Improvement of Abnormality Classification in Chest Radiographs. In 2019 41st Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC) 2019 Jul 23 (pp. 841-844). IEEE.
