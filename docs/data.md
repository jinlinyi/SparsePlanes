Data
===========



Preprocessed Data
------------------
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Size</th>
<th valign="bottom">Details</th>
<tr>
<!-- TABLE BODY -->
<td align="left"><a href="https://fouheylab.eecs.umich.edu/~jinlinyi/2021/sparsePlanesICCV21/data/rgb.zip">rgb.zip</a></td>
<td align="left">21 GB</td>
<td align="left">Habitat generated images.</td>
</tr>

 <tr><td align="left"><a href="https://fouheylab.eecs.umich.edu/~jinlinyi/2021/sparsePlanesICCV21/data/observations.zip">observations.zip</a></a></td>
<td align="left">64 GB</td>
<td align="left">Depth and semantic labels.</td>
</tr>

</tbody></table>


Dataset Json Files
------------------
We write a custom dataloader in Detectron2 and it loads jsons that contain the dataset information.
[mp3d_planercnn_json.zip][split] (160 MB) contains jsons for `train/val/test` split. Each json file stores images pairs and their annotations.
```yaml
# json file data structure
"info": "...",
"categories":  [{'id': 0, 'name': 'plane'}],
"data": [
    "0": {                                      # image A
        "file_name": /path/to/image_id.png,
        "image_id": image_id,
        "height": 480,
        "width": 640,
        "camera": {                             # camera pose in the asset
            "position": [x,y,z],                
            "rotation": [w, xi, yi, zi],        # quaternion
        }
        "annotations": [                        # list of planes, with detectron2 annotations format.
            {                                   
                "id":                           
                "image_id":
                "category_id":
                "iscrowd":
                "area": 
                "bbox":
                "bbox_mode":
                "width":
                "height":
                "segmentation":
                "plane":                        # plane parameters
            },
        ]
    },
    "1": {...},                                 # image B
    "gt_corrs": [...],                          # List of pairs of corresponding plane indices
    'rel_pose': {                               # A's pose in B's coordinate frame.
        'position':
        'rotation':
    },
    ...
]
```

[split]: https://fouheylab.eecs.umich.edu/~jinlinyi/2021/sparsePlanesICCV21/split/mp3d_planercnn_json.zip