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
<!-- TABLE BODY -->
 <tr><td align="left">Plane</a></td>
<td align="left">step1-plane.yaml</td>
</tr>

 <tr><td align="left">Embedding</a></td>
<td align="left">step2-corr.yaml</td>
</tr>

 <tr><td align="left">Camera</a></td>
<td align="left">step3-camera.yaml</td>
</tr>

</tbody></table>