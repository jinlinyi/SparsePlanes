Data
===========
Please refer to the [README][readme] in foler `data_preparation/` for details about how we preprocess data.


Preprocessed Data
------------------
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Size</th>
<th valign="bottom">Details</th>
<th valign="bottom">Required for training?</th>
<tr>
<!-- TABLE BODY -->
<td align="left"><a href="https://www.dropbox.com/s/uqkcaaoayd0me8e/mp3d_planercnn_json.zip">mp3d_planercnn_json.zip</a></td>
<td align="left">160 MB</td>
<td align="left">Jsons that contain the dataset information.</td>
<td align="center">Yes</td>
</tr>

<td align="left"><a href="https://www.dropbox.com/s/po09x0aovog9oe1/rgb.zip">rgb.zip</a></td>
<td align="left">21 GB</td>
<td align="left">Habitat generated images.</td>
<td align="center">Yes</td>
</tr>

 <tr><td align="left"><a href="https://www.dropbox.com/s/otea8zdyadmxj15/observations.zip">observations.zip</a></a></td>
<td align="left">64 GB</td>
<td align="left">Depth and semantic labels.</td>
<td align="center">Yes</td>
</tr>

 <tr><td align="left"><a href="https://www.dropbox.com/s/1cutwfqhsx30joh/id2semantic.zip">id2semantic.zip</a></a></td>
<td align="left">728 KB</td>
<td align="left">Instance id to semantic name.</td>
<td align="center">No</td>
</tr>

 <tr><td align="left"><a href="https://www.dropbox.com/s/84ulrsk47b72nfv/planes_ply_mp3dcoord_refined.zip">planes_ply_mp3dcoord_refined.zip</a></a></td>
<td align="left">28 GB</td>
<td align="left">Plane annotations.</td>
<td align="center">No</td>
</tr>

 <tr><td align="left"><a href="https://www.dropbox.com/s/ul1v2vrlzl4voxj/cameras.zip">cameras.zip</a></a></td>
<td align="left">4.4 MB</td>
<td align="left">Camera poses.</td>
<td align="center">No</td>
</tr>

</tbody></table>



Dataset Json Files
------------------
We write a custom dataloader in Detectron2 and it loads jsons that contain the dataset information.
[mp3d_planercnn_json.zip][split] contains jsons for `train/val/test` split. Each json file stores images pairs and their annotations.
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
[readme]: ../data_preparation/README.md
[split]: https://fouheylab.eecs.umich.edu/~jinlinyi/2021/sparsePlanesICCV21/split/mp3d_planercnn_json.zip
