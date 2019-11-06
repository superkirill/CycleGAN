# IPCV's Tutored Reasearch and Development Project (TRDP): Image-to-Image Translation with Conditional Adversarial Networks

Image-to-image translation task is an important problem of computer vision where the main purpose is to obtain a correspondence between two image domains. One of the most recent achievements in this field are, so called, Cycle Generative Adversarial Networks that tackle a serious issue of Supervised Learning approaches: the necessity of having paired training data. However, while CycleGANs introduce an appealing concept of utilizing unpaired training data, the mapping between two domains can be challenging and ambiguous, therefore this method may fail to converge to a global optimum. In order to combat this problem, we introduce a novel CycleGANs model. The proposed architecture allows to operate on three domains, as opposed to two  domains in the original architecture. This way, falling in local minima is less probable during the optimization stage and more information can be extracted and used to construct a more accurate representation of the target domain. The proposed model has been trained with unpaired data of three types: RGB-footage of the city, semantic segmentation and depth maps, with RGB-footage and semantic segmentation being the target domains. The obtained results prove that Semantic-to-RGB image transfer demonstrates better performance when the depth maps are incorporated in the training process.

## The architecture
Our proposal in this work is to apply the cycle consistency loss to three domains instead of two, therefore, in this new case we would have three cycles and not just one. From such alteration, we expect to obtain a potential improvement in the optimization problem, since the additional information that the third domain provides may prevent from falling in local minima.
Our original proposal was an architecture in which all the translations between domains are learned, for this, we would need a total of 6 generators and 3 discriminators. Due to the high computational cost that wold require to train a total of 9 networks, we decided to use a simplified version in which we remove the connections between the domains B and C and, therefore, remove two of the six generators. 


|![Original proposed architecture with all the connections between the three domains](https://github.com/superkirill/CycleGAN/blob/master/tricycleGAN_full.png?raw=true)  |  ![enter image description here](https://github.com/superkirill/CycleGAN/blob/master/tricycleGAN_simplified.png?raw=true)|
|--|--|
## Experiments
The experimental part of this work consisted of training the original [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) architecture to learn image translation from RGB to semantic segmentation domain. The performance of this system was taken as the baseline. Later, the proposed architecture was trained on the same data with the addition of depth domain. This provides the ground for evaluation of the proposed architecture. The dataset chosen for the experiments was the  [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/). Some results can be found in the next image. 

![enter image description here](https://github.com/superkirill/CycleGAN/blob/master/comparison.jpg?raw=true)
 ## Train and test
-   For training the model, use the following command:

    python train.py --dataroot datasets/maps --name maps_cyclegan --model cycle_gan

-   To run the test, use the following command:

    python test.py --dataroot datasets/maps --name maps_cyclegan --model cycle_gan
