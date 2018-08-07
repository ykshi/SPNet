import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image
import glob
import os, h5py

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)

class Hdf5_dataset(data.Dataset):

  def __init__(self, path_patients):
      hdf5_list = [x for x in glob.glob(os.path.join(path_patients,'*.h5'))]#only h5 files
      print 'h5 list ',hdf5_list
      self.datasets = []
      self.datasets_gt=[]
      self.total_count = 0
      self.limits=[]
      for f in hdf5_list:
         h5_file = h5py.File(f, 'r')
         dataset = h5_file['data']
         dataset_gt = h5_file['label']
         self.datasets.append(dataset)
         self.datasets_gt.append(dataset_gt)
         self.limits.append(self.total_count)
         self.total_count += len(dataset)
        #  print 'len ',len(dataset)
      #print self.limits

  def __getitem__(self, index):

      dataset_index=-1
    #   print 'index ',index
      for i in xrange(len(self.limits)-1,-1,-1):
        #print 'i ',i
        if index>=self.limits[i]:
          dataset_index=i
          break
      #print 'dataset_index ',dataset_index
      assert dataset_index>=0, 'negative chunk'

      in_dataset_index = index-self.limits[dataset_index]
    #   print 'len: ', len(self.datasets[dataset_index])

      return self.datasets[dataset_index][in_dataset_index], self.datasets_gt[dataset_index][in_dataset_index]

  def __len__(self):
      return self.total_count
