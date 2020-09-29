# Deep-Residual-Shrinkage-Networks
The deep residual shrinkage network is a variant of deep residual networks (ResNets), and aims to improve the feature learning ability from highly noise signals or complex backgrounds. Although the method is originally developed for vibration-based fault diagnosis, it can be applied to image recognition and speech recognition as well. The major innovation is the integration of soft thresholding as nonlinear transformation layers into ResNets. Moreover, the thresholds are automatically determined by a specially designed sub-network, so that no professional expertise on threshold determination is required.

![The basic idea of deep residual shrinkage networks](https://github.com/zhao62/Deep-Residual-Shrinkage-Networks/blob/master/Basic-idea-of-DRSN.png)

The method is implemented using TensorFlow 1.0.1, TFLearn 0.3.2, and Keras 2.2.1, and applied for image classification. A small network with 3 residual shrinkage blocks is constructed in the code. More blocks and more training iterations can be used for a higher performance.

Reference:
Minghang Zhao, Shisheng Zhong, Xuyun Fu, Baoping Tang, Michael Pecht, Deep residual shrinkage networks for fault diagnosis, 
IEEE Transactions on Industrial Informatics, 2020, 16(7): 4681-4690.

Abstract:
This paper develops new deep learning methods, namely, deep residual shrinkage networks, to improve the feature learning ability from highly noised vibration signals and achieve a high fault diagnosing accuracy. Soft thresholding is inserted as nonlinear transformation layers into the deep architectures to eliminate unimportant features. Moreover, considering that it is generally challenging to set proper values for the thresholds, the developed deep residual shrinkage networks integrate a few specialized neural networks as trainable modules to automatically determine the thresholds, so that professional expertise on signal processing is not required. The efficacy of the developed methods is validated through experiments with various types of noise.

https://ieeexplore.ieee.org/document/8850096
