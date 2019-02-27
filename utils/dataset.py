from __future__ import division, print_function
import os
import glob
import numpy as np
from torch.utils import data
from . import patient


def load_images(data_dir, mask='both'):
    """Load all patient images and contours from TrainingSet, Test1Set or
    Test2Set directory. The directories and images are read in sorted order.

    Arguments:
      data_dir - path to data directory (TrainingSet, Test1Set or Test2Set)

    Output:
      tuples of (images, masks), both of which are 4-d tensors of shape
      (batchsize, height, width, channels). Images is uint16 and masks are
      uint8 with values 0 or 1.
    """
    assert mask in ['inner', 'outer', 'both']

    glob_search = os.path.join(data_dir, "patient*")
    patient_dirs = sorted(glob.glob(glob_search))
    if len(patient_dirs) == 0:
        raise Exception("No patient directors found in {}".format(data_dir))

    # load all images into memory (dataset is small)
    images = []
    inner_masks = []
    outer_masks = []
    for patient_dir in patient_dirs:
        p = patient.PatientData(patient_dir)
        images += p.images
        inner_masks += p.endocardium_masks
        outer_masks += p.epicardium_masks

    # reshape to account for channel dimension
    images = np.asarray(images)[:,None,:,:]
    if mask == 'inner':
        masks = np.asarray(inner_masks)
    elif mask == 'outer':
        masks = np.asarray(outer_masks)
    elif mask == 'both':
        # mask = 2 for endocardium, 1 for cardiac wall, 0 elsewhere
        masks = np.asarray(inner_masks) + np.asarray(outer_masks)
    else :
        masks = np.asarray(inner_masks)
    # one-hot encode masks，but we use cross entropy don not need convert
    # dims = masks.shape
    # classes = len(set(masks[0].flatten())) # get num classes from first image
    # new_shape = dims + (classes,)
    # masks = utils.to_categorical(masks).reshape(new_shape)

    return images, masks

class RvscDataset(data.Dataset):

    def __init__(self, data_dir, mask, trainset = True,validation_split=0.0,normalize_images=True):

        images, masks = load_images(data_dir, mask)
        split_index = int((1 - validation_split) * len(images))
        if trainset:
            self.images = images[:split_index].astype(np.float64)
            self.masks = masks[:split_index].astype(np.float64)
        else:
            self.images = images[split_index:].astype(np.float64)
            self.masks = masks[split_index:].astype(np.float64)


        if normalize_images:
            normalize(self.images, axis=(1, 2))



    def __getitem__(self, index):
        """
        返回：
        - img: 图像
        - mask: 掩膜
        """
        img = self.images[index]
        mask = self.masks[index]
        return img, mask

    def __len__(self):
        return len(self.images)

def normalize(x, epsilon=1e-7, axis=(1,2)):
    x -= np.mean(x, axis=axis, keepdims=True)
    x /= np.std(x, axis=axis, keepdims=True) + epsilon

def get_dataloader(data_dir, batch_size, trainset = True, validation_split=0.0, mask='both', shuffle=True, normalize_images=True):

    dataset = RvscDataset(data_dir, mask,trainset, validation_split, normalize_images)

    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=0)
    return dataloader











