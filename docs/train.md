Training
========
To train a model run:
```bash
cd $PRJ_ROOT/sparsePlane
python tools/train_net.py \
--config-file configs/$CONFIG \
--num-gpus $NUM_GPU
```
we take three steps to train while freezing the previous part:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Target</th>
<th valign="bottom">$CONFIG</th>
<th valign="bottom">Pretrain</th>
<!-- TABLE BODY -->
 <tr><td align="left">Plane</a></td>
<td align="left">step1-plane.yaml</td>
<td align="left"><a href="https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md#faster-r-cnn">faster_rcnn_R_50_FPN_3x</a></td>
</tr>

 <tr><td align="left">Embedding</a></td>
<td align="left">step2-corr.yaml</td>
<td align="left"><a href="https://fouheylab.eecs.umich.edu/~jinlinyi/2021/sparsePlanesICCV21/models/step1_model_0036999.pth">step1_model_0036999.pth</a></td>
</tr>

 <tr><td align="left">Camera</a></td>
<td align="left">step3-camera.yaml</td>
<td align="left"><a href="https://fouheylab.eecs.umich.edu/~jinlinyi/2021/sparsePlanesICCV21/models/step2_model_0035999.pth">step2_model_0035999.pth</a></td>
</tr>

</tbody></table>