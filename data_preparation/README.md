# Training data generation from Matterport3D dataset.
We augment Matterport3D with plane segmentation 
annotations on the original mesh. This enables consistent
plane instance ID and plane parameters through rendering,
and automatically establishes plane correspondences across
different images. We fit planes using RANSAC, following [PlaneNet][4], on mesh vertices within the same object instance
(using the annotation provided by the original Matterport3D
dataset). We then render the per-pixel plane instance ID,
along with the RGB-D images using AI Habitat [habitat-sim][3]. Since
the original instance segmentation mesh has “ghost” objects
and holes due to artifacts, we further filter out bad plane annotations by comparing the depth information and the plane
parameters. 
## Fit planes on Matterport3d meshes
You need to download [Matterport3D][1] dataset and [Habitat-MP3D][2]. 
We heavily used code from [PlaneNet][4] to fit planes. 
```
# Fit plane
bash fit_plane_ply.bash

# Generate plane segment mesh
bash convert_mp3d_ply.bash
```

## Render gt plane segmentation
Note: you need to install [habitat-sim][3].
```bash
bash render_plane.bash
```

[1]: https://niessner.github.io/Matterport/
[2]: https://github.com/facebookresearch/habitat-lab#Matterport3D
[3]: https://github.com/facebookresearch/habitat-sim
[4]: https://github.com/art-programmer/PlaneNet/tree/master/data_preparation