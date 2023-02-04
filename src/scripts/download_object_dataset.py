import urllib.request
import zipfile
import os
import json
import argparse
import shutil
from tqdm import tqdm

from src.scripts.process_clouds import process_grasp_ds

# https://stackoverflow.com/a/53877507/20436693
class DownloadProgressBar(tqdm):
    """ 
    A class to show a progress bar when downloading a file 
    
    Methods
    -------
    update_to(b=1, bsize=1, tsize=None)
        Update the progress bar
    """
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """ 
    Download a file from a given url and save it to a given path

    Parameters
    ----------
    url : str
        The url to download the file from
    output_path : str
        The path to save the downloaded file to
    """
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc='Dataset version '+url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def parse_args():
    """
    Parse input arguments.
    Args are:
        grasp: whether to download the grasp dataset
        object: whether to download the object dataset

    Returns
    -------
    args : argparse.Namespace
        The parsed arguments
    """

    # parse argument to check whether to download the grasp dataset, object dataset or both 
    # Also check whether to process grasp dataset when downloaded
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--grasp', action='store_true', default=False, help='Download the grasp dataset')
    parser.add_argument('--object', action='store_true', default=False, help='Download the object dataset')
    parser.add_argument('--process_grasp', action='store_true', default=False, help='Process the grasp dataset')
    args = parser.parse_args()

    return args

def main(cfg):
    """
    Download the grasp and object datasets and process the grasp dataset.
    What is done depends on the args passed to the script.

    Parameters
    ----------
    cfg : dict
        Config dictionary
    """

    args = parse_args()

    # download grasp dataset
    if args.grasp:
        grasp_ds_pth = cfg['dirs']['file_dir'] + cfg['dirs']['grasp_data_dir']
        if not os.path.exists(grasp_ds_pth):
            os.makedirs(grasp_ds_pth)
        
        # download 6dof robotic grasp ds from monash bridges
        # https://bridges.monash.edu/articles/dataset/6-DoF_Real_Robotic_Grasping_Dataset/20174165
        download_url("https://bridges.monash.edu/ndownloader/articles/20174165/versions/4", grasp_ds_pth+'data.zip')

        # unzip data
        with zipfile.ZipFile(grasp_ds_pth+'data.zip', 'r') as zip_ref:
            print('Unzipping grasp dataset...')
            zip_ref.extractall(grasp_ds_pth)
        os.remove(grasp_ds_pth+'data.zip')

        # unzip all the unzipped zip files
        for file in os.listdir(grasp_ds_pth):
            if file.endswith(".zip"):
                with zipfile.ZipFile(grasp_ds_pth+file, 'r') as zip_ref:
                    print('Unzipping '+file)
                    zip_ref.extractall(grasp_ds_pth)
                os.remove(grasp_ds_pth+file)
                shutil.rmtree(grasp_ds_pth+'__MACOSX/')
        
        # make point_clouds folder
        if not os.path.exists(grasp_ds_pth+'point_clouds/'):
            os.makedirs(grasp_ds_pth+'point_clouds/')
        # move all files in point_clouds part folders into a point_clouds folder
        for file in os.listdir(grasp_ds_pth):
            if 'point_clouds_' in file:
                for file2 in os.listdir(grasp_ds_pth+file):
                    shutil.move(grasp_ds_pth+file+'/'+file2, grasp_ds_pth+'point_clouds/')
                shutil.rmtree(grasp_ds_pth+file)

        if args.process_grasp:
            process_grasp_ds(cfg)


    # download object dataset
    if args.object:
        object_ds_path =  cfg['dirs']['file_dir']  + cfg['dirs']['object_data_dir'] 
        if not os.path.exists(object_ds_path):
            os.makedirs(object_ds_path)

        # download object ds from monash bridges
        # https://bridges.monash.edu/articles/dataset/Supermarket_Object_Dataset/20179550
        download_url('https://bridges.monash.edu/ndownloader/articles/20179550/versions/2', object_ds_path+'data.zip')

        # unzip data
        with zipfile.ZipFile(object_ds_path+'data.zip', 'r') as zip_ref:
            print('Unzipping object dataset...')
            zip_ref.extractall(object_ds_path)
        os.remove(object_ds_path+'data.zip')

        # unzip all the unzipped zip files
        for file in os.listdir(object_ds_path):
            if file.endswith(".zip"):
                with zipfile.ZipFile(object_ds_path+file, 'r') as zip_ref:
                    print('Unzipping '+file)
                    zip_ref.extractall(object_ds_path)
                os.remove(object_ds_path+file)

if __name__ == '__main__':
    # load config file
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    main(config)