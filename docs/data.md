Data
===========
We store our data in [Dropbox][1].


Dataset Json Files
------------------
We write a custom dataloader in Detectron2 and it loads jsons that contain the dataset information.
`$DROPBOX_FOLDER/mp3d_planercnn_json/` contains jsons for `train/val/test` split. Each json file stores images pairs and their annotations.
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

Preprocessed Data
------------------
`$DROPBOX_FOLDER/data/` contains all the preprocessed data. You need to unzip `rgb.zip` and `observations.zip` if you want to re-train the network.


[1]: https://www.dropbox.com/sh/bfafx8vz5pmy196/AADU1qZmjbuZzEiNzeqGmBala