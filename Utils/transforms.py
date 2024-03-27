''''Library for 3D [Height, Width, Depth] volume transformations.
'''

# import dependency library
import random
#import collections
import collections.abc
import torch
import numpy as np
from scipy import ndimage
from skimage.transform import resize
from scipy.ndimage.morphology import distance_transform_edt


# ===================intereface of transformation(augmentation)  method(classes)=====================
class Base(object):
    """
    #=======================================
    #    Basic class for transformation
    #=======================================
    """
    #  sample random variables
    def sample(self, *shape):
        return shape

    #  define transformation actions
    def tf(self, image, image_name, k=0):  # By default, img is formatted as list as [raw, target]
        ## If reuse is set as True, no sampling is needed
        return image

    # define how to call tf, all the subclass will have this function
    # they can be called, and then execute the override tf function
    def __call__(self, image_list, image_name,dim=3, reuse_pros=False):  #
        # resampling parameters and set self properties
        if not reuse_pros:
            raw_im = image_list if isinstance(image_list, np.ndarray) else image_list[0] # read raw image
            shape = raw_im.shape
            assert len(shape) == 3, "only support 3-dim data"
            self.sample(*shape)

        if isinstance(image_list, collections.abc.Sequence): # raw_mem, labeled_mem, labeled_nuc
            return [self.tf(x, image_name,k) for k, x in enumerate(image_list)]
        else:
            return self.tf(image_list,image_name)

    #  define print string
    def __str__(self):
        return 'Identity()' # used for eval()

Identity = Base
# ===========================================================================================

# ==================rotation transformation(augmentation)================================
#=====================================
#      Rotation
#=====================================
class Rot90(Base):
    def __init__(self, axes=(0, 1)):
        self.axes = axes
        for a in self.axes:
            assert a > -1

    def sample(self, *shape):
        shape0 = list(shape)
        shape0[self.axes[0]] = shape[self.axes[1]]
        shape0[self.axes[1]] = shape[self.axes[0]]
        return shape0

    def tf(self, image, image_name, k=0):
        # print('rotating 90 degree for image of ',image_name)
        return np.rot90(image, axes=self.axes)

    def __str__(self):
        return "Rot90(axes=({}, {}))".format(*self.axes)


class RandomRotation(Base):
    def __init__(self, angle_spectrum=10):
        assert isinstance(angle_spectrum,int)
        self.angle_spectrum = angle_spectrum
        self.axes = [(1, 0), (2, 1), (2, 0)]

    def sample(self, *shape):
        self.axes_buffer = self.axes[np.random.choice(list(range(len(self.axes))))]
        self.angle_buffer = np.random.randint(-self.angle_spectrum, self.angle_spectrum)
        return list(shape)

    def tf(self, image, image_name, k=0):  #
        # print('random rotating for image of ',image_name)
        image = ndimage.rotate(image, self.angle_buffer, axes=self.axes_buffer, reshape=False, order=0, mode="constant", cval=0)
        return image

    def __str__(self):
        return "RandomRotion(axes={}, angle={}".format(self.axes_buffer, self.angle_buffer)
# ===========================================================================================


#===========================================
#   Flip
#===========================================
class Flip(Base):
    def __init__(self, axis=0):
        self.axis = axis

    def tf(self, image, image_name, k=0):
        print('numpy flip for image of ',image_name)
        return np.flip(image, self.axis)

    def __str__(self):
        return "Flip(axis={})".format(self.axis)

class RandomFlip(Base):
    def __init__(self, axis=0):
        self.axis = (0, 1, 2)
        self.x_buffer = None
        self.y_buffer = None
        self.z_buffer = None

    def sample(self, *shape):
        self.x_buffer = np.random.choice([True, False])
        self.y_buffer = np.random.choice([True, False])
        self.z_buffer = np.random.choice([True, False])
        return list(shape)

    def tf(self, image, image_name, k=0):
        # print('(data augmentation)random flip for image of ',image_name)
        if self.x_buffer:
            image = np.flip(image, axis=self.axis[0])
        if self.y_buffer:
            image = np.flip(image, axis=self.axis[1])
        if self.z_buffer:
            image = np.flip(image, axis=self.axis[2])

        return image

    def __str__(self):
        return "Random Flip images with Flip x,y,z : {} {} {}".format(str(self.x_buffer),str(self.y_buffer),str(self.z_buffer))


#============================================
#   crop when extend the volume
#============================================
class CenterCrop(Base):
    def __init__(self, target_size):
        self.target_size = target_size
        self.buffer = None

    def sample(self, *shape):
        target_size = self.target_size
        start = [(rs - ts)//2 for rs, ts in zip(shape, target_size)]
        self.buffer = [slice(st, st + ts) for st, ts in zip(start, target_size)]
        # print(self.buffer)

        return target_size

    def tf(self, image, image_name, k=0):
        # print('crop center 3d images of ', image_name)
        return image[tuple(self.buffer)]

    def __str__(self):
        return "CenterCrop({})".format(self.target_size)


class RandCrop(CenterCrop):
    def sample(self, *shape):
        # print(start,shape,self.target_size)

        start = [random.randint(0, rs - ts) for rs, ts in zip(shape, self.target_size)]
        # print(start,shape,self.target_size)
        self.buffer = [slice(st, st + ts) for st, ts in zip(start, self.target_size)]
        # print(self.buffer)
        return self.target_size

    def __str__(self):
        return "Random Crop({})".format(self.target_size)


#======================================================
#   Pad when smaller than the target
#======================================================
class Pad(Base):
    def __init__(self, target_size=[208, 288, 144]):
        self.target_size = target_size

    def sample(self, *shape):
        return self.target_size

    def tf(self, image, image_name,k=0):
        raw_shape = image.shape
        pad_slice = [i - j for i, j in zip(self.target_size, raw_shape)]
        self.px = tuple((i//2, i-i//2) for i in pad_slice)
        return np.pad(image, self.px, mode="constant")

    def __str__(self):
        return "Pad {}, {}, {}".format(*self.px)


#===========================================
#   change intensity
#===========================================
class RandomIntensityChange(Base):
    def __init__(self, factor):
        # print('RandomIntensityChange transformation init with {} factor'.format(factor))
        shift, scale = factor
        assert (shift > 0) and (scale > 0), "shift {} and scale {} must > 0".format(shift, scale)
        self.shift = shift
        self.scale = scale

    def tf(self, image, image_name,k=0):
        # print('RandomIntensityChange transformation execution with {} shift {} scale'.format(self.shift,self.scale))
        if k==0:
            shift_buffer = np.random.uniform(-self.shift, self.shift, size = list(image.shape))
            scale_factor = np.random.uniform(1.0 - self.scale, 1.0 + self.scale, size=list(image.shape))
            return image * scale_factor + shift_buffer
        else:
            return image

    def __str__(self):
        return 'RandomIntensityChange transformation with {} shift {} scale'.format(self.shift,self.scale)


# =========================add noise==================================
class Noise(Base):
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def tf(self, image, image_name,k=0):
        if k == 1:
            return image
        shape = image.shape
        return image * np.exp(self.sigma * torch.randn(shape, dtype=torch.float32).numpy())

    def __str__(self):
        return "Noise()"



# ===============================================================================
#   binary mask to distance
class ContourEDT(Base):
    #  only applicable to binary MembAndNuc
    def __init__(self, edt_threshold=15):
        # print('ContourEDT transformation init with {} edt_threshold'.format(edt_threshold))
        self.edt_threshold = edt_threshold

    def tf(self, image,image_name, k=0):
        # print('ContourEDT transformation execution with {} edt_threshold'.format(self.edt_threshold))
        if k==1 and len(np.unique(image)) == 2: # make sure binary image
            background_edt = distance_transform_edt(image == 0)  # calculate the distance from backgroud to labelmembrane
            # print(np.unique(background_edt.astype(int), return_counts=True))
            background_edt[background_edt > self.edt_threshold] = self.edt_threshold
            norm_mem_edt = (self.edt_threshold - background_edt) / self.edt_threshold  # the normalize distance from labelmembrane to backgroud

            return norm_mem_edt.astype(np.float32)

        else:
            return image
    def __str__(self):
        return 'ContourEDT transformation with {} edt_threshold'.format(self.edt_threshold)


#   Normalization
class Normalize(Base):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def tf(self, image,image_name, k=0):
        if k==1:
            return image
        image -= self.mean
        image = self.std
        return image

    def __str__(self):
        return "Normalize()"


class Resize(Base):
    def __init__(self, target_size=(205, 288, 144)):
        assert len(target_size) == 3, "Only support in-slice resize"  #
        self.target_size = target_size

    def tf(self, image, image_name,k=0):
        if k == 0:
            resized_stack = resize(image, self.target_size, mode='constant', cval=0, order=1, anti_aliasing=True)
        else:
            resized_stack = resize(image, self.target_size, mode='constant', cval=0, order=0, anti_aliasing=False)
        return resized_stack


#=============================================
#   randomly select operations from a series of operations
#=============================================
class RandSelect(Base):
    def __init__(self, prob=0.5, tf=None):
        self.prob = prob
        self.ops = tf if isinstance(tf, collections.abc.Sequence) else (tf, )
        self.buff = None

    def sample(self, *shape):
        self.buff = random < self.prob

        if self.buff:
            for op in self.ops:
                shape = op.sample(*shape)
        return shape

    def tf(self, image, iamge_name,k=0):
        if self.buff:
            for op in self.ops:
                image = op.tf(image, k)
        return image

    def __str__(self):
        if len(self.ops) == 1:
            ops = str(self.ops[0])
        else:
            ops = "[{}]".format(','.join([str(op) for op in self.ops]))
        return "RandSelect({}, {})".format(self.prob, ops)



#=================================================
#   Data format transformation
#=================================================
class ToNumpy(Base):
    def __init__(self):
        pass

    def tf(self, image,image_name, k=0):
        return image.numpy()

    def __str__(self):
        return "ToNumpy()"


class ToTensor(Base):
    def __init__(self):
        pass

    def tf(self, image,image_name, k=0):
        return torch.from_numpy(image)

    def __str__(self):
        return "Transformation to torch tensor"


class TensorType(Base):
    def __init__(self, types):
        self.types = types

    def tf(self, image, image_name,k=0):
        return image.type(self.types[k])

    def __str__(self):
        s_types = ",".join([str(s) for s in self.types])
        return "Transformation to tensorType(({}))".format(s_types)


class NumpyType(Base):
    def __init__(self, types):
        self.types = types

    def tf(self, image, image_name,k=0):
        # print('image as numpy type, ', self.types[k])
        return image.astype(self.types[k])  #

    def __str__(self):
        s_types = ",".join([str(s) for s in self.types])
        return "Transformation to NumpyType(({}))".format(s_types)


# ================ execute a list of subclass (augmentation transformation) of Identity(Base)==========
class Compose(Base):
    """
    raw data augmentation sequence
    #=================================================
    #   Compose different operations.
    #=================================================
    """
    def __init__(self, ops):
        if not isinstance(ops, collections.abc.Sequence):
            ops = ops,
        self.ops = ops

    def sample(self, *shape):
        for op in self.ops:
            # print('Composing func sample ', op)
            shape = op.sample(*shape)

    def tf(self, image, image_name,k=0):
        #is_tensor = isinstance(img, torch.Tensor)
        #if is_tensor:
        #    img = img.numpy()

        for op in self.ops:
            # print('========>>>>>>>')
            # print('Transformation of ', op)
            # print('input image shape',image.shape, ' type: ', type(image))
            # print('embryo name',image_name)
            # print('k value:',k)

            image = op.tf(image, image_name,k) # do not use op(img) here

            # print('output image shape',image.shape, ' type: ', type(image))
            # print('===============')


        #if is_tensor:
        #    img = np.ascontiguousarray(img)
        #    img = torch.from_numpy(img)
        return image

    def __str__(self):
        ops = ', '.join([str(op) for op in self.ops])
        return 'Compose([{}])'.format(ops)
# ===============================================================================================


