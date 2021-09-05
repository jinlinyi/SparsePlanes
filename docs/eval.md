# Evaluation
## Final results
Final results are saved [here][1]. 
You can download and extract it to `$PRJ_ROOT/sparsePlane/results`.

Alternatively, you can generate the results by:
```bash
# To evaluate AP, camera
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/train_net.py \
--config-file tools/demo/config.yaml \
--eval-only \
--num-gpus 4 \
DATASETS.TEST "('mp3d_test',)" \
MODEL.WEIGHTS ./models/model_ICCV.pth \
OUTPUT_DIR ./results/predbox

# For correspondence, we use GT box.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/train_net.py \
--config-file tools/demo/config.yaml \
--eval-only \
--num-gpus 4 \
DATASETS.TEST "('mp3d_test',)" \
MODEL.WEIGHTS ./models/model_ICCV.pth \
TEST.EVAL_GT_BOX True \
OUTPUT_DIR ./results/gtbox
```

## Evaluate AP
```bash
python tools/eval.py \
--config-file results/predbox/config.yaml \
--rcnn-cached-file results/predbox/instances_predictions.pth \
--camera-cached-file results/predbox/summary.pkl \
--optimized-dict-path results/predbox/continuous.pkl \
--evaluate AP 
```

## Evaluate Camera
```bash
python tools/eval.py \
--config-file results/predbox/config.yaml \
--rcnn-cached-file results/predbox/instances_predictions.pth \
--camera-cached-file results/predbox/summary.pkl \
--optimized-dict-path results/predbox/continuous.pkl \
--evaluate camera
```

## Evaluate Correspondence
```bash
python tools/eval.py \
--config-file results/gtbox/config.yaml \
--rcnn-cached-file results/gtbox/instances_predictions.pth \
--camera-cached-file results/gtbox/summary.pkl \
--evaluate correspondence \
--optimized-dict-path results/gtbox/discrete.pkl
```

## Note
If you do not want to use cached results, set `--optimized-dict-path` to be `''`, then `eval.py` will generate `optimized_dict` online.
In `eval.py`, you can uncomment 
`save_dict(optimized_dict, './results/gtbox', 'discrete')` to save `discrete.pkl`.

[1]: https://fouheylab.eecs.umich.edu/~jinlinyi/2021/sparsePlanesICCV21/results.zip