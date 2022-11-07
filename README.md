**WIP** object and grasping dataset

## Real World Object and Grasping Dataset

Code for `PAPER`.

### Data

Data can be either directly downloaded from Monash Bridges or using the provided python script.

**Python**: Once the repo has been cloned, from the root directory run:

```
python scripts/download_object_dataset.py  --object --grasp --process_grasp 
```
This will download both the object and grasping datasets, and preprocess the grasping dataset. If you wish to not download one of the datasets or not to preprocess the dataset, please delete the corresponding arguments:
```
- download object dataset argument: --object
- download grasp dataset argument: --grasp
- preprocess grasp dataset argument: --process_grasp
```

**Monash Bridges**: Datasets can be downloaded from the following links (Can easily be downloaded by selecting 'Download all' on the webpages):

[Object dataset](https://bridges.monash.edu/articles/dataset/Supermarket_Object_Dataset/20179550)

[Grasping dataset](https://bridges.monash.edu/articles/dataset/6-DoF_Real_Robotic_Grasping_Dataset/20174165)

To preprocess the grasping dataset, ensure the grasping dataset is modified to have the following file structure:
```
project
|
|--grasp_ds
|  |--json_files
|  |  |--[1-20]
|  |  |  |--[0-74]
|  |--depth_images
|  |  |--[1-20]
|  |  |  |--[0-74].png
|  |--rgb_images
|  |  |--[1-20]
|  |  |  |--[0-74].png
|  |--point_clouds
|  |  |--[1-20]
|  |  |  |--[0-74].pcd
```
where the folders named 1 to 20 in point_clouds_00X have been moved into point_clouds. Then run:
```
python scripts/process_clouds.py
```

### Methods

* To load some sample data, run 
    * `python scripts/data_loading_tools.py`
* To visualise the point cloud through the data processing process, run 
    * `python scripts/visualise_data_processing.py`
* To visualise the grasp pose, run 
    * `python scripts/display_grasp_pose.py`
* To preform registration between a point cloud and it's corresponding object mesh, run 
    * `python scripts/grasp_pcl_object_registration.py`

### Model training 
WIP