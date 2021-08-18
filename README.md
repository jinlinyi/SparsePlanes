Planar Surface Reconstruction From Sparse Views
================================================================

<h4>
Linyi Jin, Shengyi Qian, Andrew Owens, David F. Fouhey
</br>
<span style="font-size: 14pt; color: #555555">
University of Michigan
</span>
</br>
ICCV 2021 (Oral)
</h4>
<hr>

This repo contains code for our [paper][1]. Our model is implemented in [Detectron2][0].

Given two RGB images with an unknown relationship, our system produces a single, coherent planar surface reconstruction of the scene in terms of 3D planes and relative camera poses. 

![model-architecture](docs/static/architecture.jpg)
We use a ResNet50-FPN to detect planes and predict probabilities of relative camera poses, and use a two-step optimization to generate a coherent planar reconstruction. (a) For each plane, we predict a segmentation mask, plane parameters, and an appearance feature. (b) Concurrently, we pass image features from the detection backbone through the attention layer and predict the camera transformation between views. (c) Our discrete optimization fuses the prediction of the separate heads to select the best camera pose and plane correspondence. (d) Finally, we use continuous optimization to update the camera and plane parameters.

Usage Instructions
------------------

1. [How to setup your environment?][2]
2. [How to inference the code on a pair of images?][3]
3. [How to process the dataset?][4]
4. [How to train your model?][5]
5. [How to evaluate your model? (TODO)][6]


Citation
--------
If you find this code useful, please consider citing:

```text
@inproceedings{jin2021planar,
      title={Planar Surface Reconstruction from Sparse Views}, 
      author={Linyi Jin and Shengyi Qian and Andrew Owens and David F. Fouhey},
      booktitle = {ICCV},
      year={2021}
}
```
[0]: https://github.com/facebookresearch/detectron2
[1]: http://arxiv.org/abs/2103.14644
[2]: ./docs/environment.md
[3]: ./docs/demo.md
[4]: ./docs/data.md
[5]: ./docs/train.md
[6]: ./docs/eval.md

Acknowledgment
--------------
We thank Dandan Shan, Mohamed El Banani, Nilesh Kulkarni, Richard Higgins for helpful discussions.
Toyota Research Institute ("TRI") provided funds to assist the authors with their research but 
this article solely reflects the opinions and conclusions of its authors and not TRI or any other Toyota entity.
