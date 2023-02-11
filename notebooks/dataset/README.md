# Dataset 

Here we have a small dataset for evaluate point-e single/multi view/s. We feched the data from ShapeNet and ModelNet collecting the meshes of one object for each category.

## Getting Started & Installing

You can download the dataset from this [link](https://drive.google.com/file/d/10eTbweARlVwrvAMS6JgksFWMLzAAbswU/view?usp=sharing). 

### Executing program

Everything is collected in a zip file. You can extract the file through the command *unzip*
```
unzip dataset-modelnet-shapenet-oc.zip -d /path/to/directory
```

## Description

Hence for each collected mesh, we sampled it into a uniform point cloud and we automatically rendered from the meshes of the object multiple images from different views (10/4 views). You can see the pipeline here [views_render](https://github.com/r1cc4r2o/point-e/blob/main/notebooks/pointrender.ipynb) with all the steps .

To illustate the tree of the directories after we unzipped the file dataset-modelnet-shapenet-oc.zip:
```
<directories>
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
    > modelnet-shapenet
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

```
### Description of the files

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

        

### Dependencies


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

## Future Developments
As possible future developments, we could:
- extend the dataset with more than one object for each category
- extend the dataset with the objects in shapenet psr
- consider rendering the multiple views images of the objects with random camera pose angle

## Authors
+ riccardo tedoldi [@riccardotedoldi](https://www.instagram.com/riccardotedoldi/)
+ diego calanzone [@diegocalanzone](https://it.linkedin.com/in/diegocalanzone)

## Version History

* 0.1
    * Initial Release

## License

Copyright 2023

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Acknowledgments

* Dataset [ModelNet40](http://modelnet.cs.princeton.edu/ModelNet40.zip)
* Dataset [ShapeNetCore](https://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v2.zip)
* [Shapenet](https://shapenet.org/)
* [Modelnet](https://modelnet.cs.princeton.edu/)
* Dataset [Shapenet_psr](https://s3.eu-central-1.amazonaws.com/avg-projects/shape_as_points/data/shapenet_psr.zip)