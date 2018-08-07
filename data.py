from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from dataset import DatasetFromFolder, Hdf5_dataset

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def download_bsd300(dest="dataset"):
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        makedirs(dest)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


def get_training_set(upscale_factor,dataset):
    # root_dir = download_bsd300()
    # if dataset == 1:
    #     root_dir = '/mnt/lustre/ganyukang/Data_t1/shiyukai/data/SR/DIV2K_patch_84/'
    # else:
    #     root_dir = download_bsd300()
    root_dir = dataset
    train_dir = join(root_dir, "train/")
    crop_size = calculate_valid_crop_size(56, upscale_factor)
    
    # return Hdf5_dataset(train_dir)
    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))

def get_test_set(upscale_factor,dataset):
    # if dataset == 1:
    #     root_dir = '/mnt/lustre/ganyukang/Data_t1/shiyukai/data/SR/DIV2K_patch_84/'
    # else:
    #     root_dir = download_bsd300()
    root_dir = dataset
    test_dir = join(root_dir, "test/")
    crop_size = calculate_valid_crop_size(56, upscale_factor)

    # return Hdf5_dataset(test_dir)
    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))
