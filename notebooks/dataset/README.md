### Dataset 
To make a comparison between ours multi views with the one single view, 
we generated these datasets.
Through experimentation, we generated several datasets from the 
available sources ModelNet40, ShapeNetV2, ShapeNet. 
Specifically, the datasets generated from ModelNet40, ShapeNetV2 are 
without textures because they were not present in the source files.
Thus, where they were not present we added textures using two approaches 
that considered the position of the point in space.
In ShapeNet the textures were present, so all views of the generated 
objects have the ground truth texture.

#### Getting Started & Installing

This is the [link](https://drive.google.com/drive/folders/1qPWA2J4e08tErD8720NlPISAiO3hGaEq?usp=share_link) to the drive folder with all the datasets.

Source to the evaluation datasets:

| Name                                | Samples | Source        |
|-------------------------------------|---------|---            |
| ModelNet40, textureless             | 40      | [Google Drive](https://drive.google.com/file/d/1cP1-fHiSm5eOG60m5WwU08U2Uz9eYp3L/view?usp=share_link)             |   
| ShapeNetv2, textureless             | 55      | [Google Drive](https://drive.google.com/file/d/1O-htsw9h2MKLpyVlog_672iMHp_VoBcf/view?usp=share_link)             |   
| Mixed, textureless                  | 190     | [Google Drive](https://drive.google.com/file/d/1YyRBEmpot2JsC2tfeqjwBV1WJh9YMTJT/view?usp=share_link)             | 
| Shapenet with textures              | 650     | [Google Drive](https://drive.google.com/file/d/1NS6oDLRMAAHfnvVmT69y6SucDgwllCiw/view?usp=share_link)             | 
| OpenAI seed imgs/clouds             | /       | [Google Drive](https://openaipublic.azureedge.net/main/point-e/banner_pcs.zip)             |   
| OpenAI, COCO CLIP R-Precision evals | /       | [Google Drive](https://openaipublic.azureedge.net/main/point-e/coco_images.zip)             |   

Here the [link](https://drive.google.com/file/d/1UOT9GfpiNfmVJPVrG2SnMwkpXic3g9iq/view?usp=share_link) of the generated cloud 
from the dataset ShapeNetv2 and ModelNet40 textureless 
comprehensive of the ground truth data, score and plot of the pairwise divergence distribution. More 
details are provided in the description.

#### Executing program

Everything is collected in a zip file. You can extract the file through the command *unzip*
```
unzip nameofthefile.zip -d /path/to/directory
```

#### Description

Hence for each collected mesh found, we sampled it into a uniform point cloud. Additionally, 
we automatically rendered from the meshes of the ground truth object multiple images from 
different poses. 

Concerning the set of views in the dataset produced from ShapeNetv2 and ModelNet40 textureless:
+ the light of the scene is fixed
+ each view is without any reflections
+ we fixed the elevation and the distance of the camera from the object and we 
took 4 pictures rotating around the object

You can see the pipeline for the generation of the ShapeNetv2 and ModelNet40 textureless dataset here 
[views_render](https://github.com/r1cc4r2o/point-e/blob/main/notebooks/1_renderTheViews_withoutTextures.ipynb) 
with all the steps.

Concerning the set of views in the dataset produced from ShapeNet with texture:
+ the light of the scene is fixed
+ each view is without any reflections
+ we develop two versions of the dataset:
  + we fixed the elevation and the distance of the camera from the object and we 
  took 6 pictures rotating around the object
  + we fixed the distance of the camera from the object and we 
  took 6 pictures changing stochastically the value of the elevation
  of the camera and rotating around the object 

You can see the pipeline for the generation of the ShapeNet dataset with textures here 
[views_render](https://github.com/r1cc4r2o/point-e/blob/main/notebooks/1_renderTheViews_withTextures.ipynb) 
with all the steps.


Here is shown the tree the [directories](https://drive.google.com/drive/folders/1qPWA2J4e08tErD8720NlPISAiO3hGaEq?usp=share_link) with the files: 
```
<directories>
    > shapenet_withTextures
        >> eval_clouds.pickle
        >> eval_views_fixed_elevation.pickle
        >> eval_views_stochastic_elevation.pickle       
    > modelnet40_texrand_texsin
        >> modelnet_csinrandn
            >>> CLASS_MAP.pt
            >>> images_obj.pt
            >>> labels.pt
            >>> points.pt
        >> modelnet_texsin
            >>> CLASS_MAP.pt
            >>> images_obj.pt
            >>> labels.pt
            >>> points.pt
    > shapenetv2_texrand_texsin
        >> shapenetv2_csinrandn
            >>> CLASS_MAP.pt
            >>> images_obj.pt
            >>> labels.pt
            >>> points.pt
        >> shapenetv2_texsin
            >>> CLASS_MAP.pt
            >>> images_obj.pt
            >>> labels.pt
            >>> points.pt
    > shapenetv2_modelnet40_texrand_texsin
        >> shapenet_modelnet_singleobject
            >>> modelnet_csinrandn
                >>>> CLASS_MAP.pt
                >>>> images_obj.pt
                >>>> labels.pt
                >>>> points.pt
            >>> modelnet_texsin
                >>>> CLASS_MAP.pt
                >>>> images_obj.pt
                >>>> labels.pt
                >>>> points.pt
            >>> shapenet_csinrandn
                >>>> CLASS_MAP.pt
                >>>> images_obj.pt
                >>>> labels.pt
                >>>> points.pt
            >>> shapenet_texsin
                >>>> CLASS_MAP.pt
                >>>> images_obj.pt
                >>>> labels.pt
                >>>> points.pt
    > dataset_shapenet_modelnet_texsin_withgeneratedcloud
        >> modelnet_texsin
            >>> CLASS_MAP.pt
            >>> eval_clouds_modelnet_300M.pickle
            >>> images_obj.pt
            >>> labels.pt
            >>> modelnet_gencloud_300M
            >>> points.pt
        >> shapenet_texsin
            >>> CLASS_MAP.pt
            >>> eval_clouds_shapenet_300M.pickle
            >>> images_obj.pt
            >>> labels.pt
            >>> shapenet_gencloud_300M
            >>> points.pt


```
##### Description of the files

    - dictionary with {index: 'typeOfObject'}: CLASS_MAP.pt 

    - multiple viwes for each object: images_obj.pt

    - label for each object: labels.pt

    - ground truth point cloud: points.pt

    - tensor with the the generated pointcloud with point-e 300M: modelnet_gencloud_300M, shapenet_gencloud_300M

    - dictionaries: eval_clouds_modelnet_300M.pickle, eval_clouds_shapenet_300M.pickle

        
        dictionary['nameOfTheObject'][index]

                                      index 0: divergence_ground_single
                                      index 1: divergence_ground_single_distribution_plot
                                      index 2: divergence_ground_multi
                                      index 3: divergence_ground_multi_distribution_plot
                                      index 4: divergence_single_multi
                                      index 5: divergence_single_multi_distribution_plot
                                      index 6: ground_truth_pis
                                      index 7: single_view_pis
                                      index 8: multi_view_pis
                                      index 9: ground_truth_point_cloud
                                      index 10: single_view_point_cloud
                                      index 11: multi_view_point_cloud

        

#### Dependencies


* import the files pt with torch
```
images_obj_views = torch.load(os.path.join(base_path,'images_obj.pt'))
```

* import the pickle file with the metrics
* more info in the [notebook1](https://github.com/r1cc4r2o/point-e/blob/main/notebooks/pointStat.ipynb) or [notebook2](https://github.com/r1cc4r2o/point-e/blob/main/notebooks/statPlot.ipynb).
```
dataset = 'shapenet'
base_path = os.path.join(dataset+"_texsin")
with open(os.path.join(base_path, 'eval_clouds_'+dataset+'_300M.pickle'), 'rb') as handle:
    data = pickle.load(handle)
```

#### Future Developments
As possible future developments, we could:
- extend the dataset with more than one object for each category
- extend the dataset with the objects in shapenet psr
- consider rendering the multiple views images of the objects with random camera pose angle

#### Authors
+ diego calanzone [@diegocalanzone](https://it.linkedin.com/in/diegocalanzone)
+ riccardo tedoldi [@riccardotedoldi](https://www.instagram.com/riccardotedoldi/)
#### Version History
* 0.1
    * Initial Release
### License

Copyright 2023

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#### Acknowledgments

* Dataset [ModelNet40](http://modelnet.cs.princeton.edu/ModelNet40.zip)
* Dataset [ShapeNetCore](https://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v2.zip)
* [Shapenet](https://shapenet.org/)
* [Modelnet](https://modelnet.cs.princeton.edu/)
* Dataset [Shapenet_psr](https://s3.eu-central-1.amazonaws.com/avg-projects/shape_as_points/data/shapenet_psr.zip)