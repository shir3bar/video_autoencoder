# Transforms, adapted from the pytorch documentation:
# https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html
import cv2
import numpy as np
import torch
from skimage import exposure
import PIL
import random


class RandomVerticalFlip():
    """Vertically flip an array of given images aka clip randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Randomly flip the clip vertically
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Randomly flipped clip
        """

        clip = sample['clip']
        if random.random() < self.p:
            if isinstance(clip, np.ndarray):
                frames=[np.flipud(img) for img in clip]
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                ' but got list of {0}'.format(type(clip)))
            clip = np.stack(frames)
        return {'clip': clip}

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'


class RandomHorizontalFlip():
    """Horizontally flip an array of given images aka clip randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Randomly flip the clip horizontally
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Randomly flipped clip
        """
        clip = sample['clip']
        if random.random() < self.p:
            if isinstance(clip, np.ndarray):
                frames = [np.fliplr(img) for img in clip]
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                ' but got list of {0}'.format(type(clip)))
            clip = np.stack(frames)
        return {'clip': clip}

    def __repr__(self):
        return f'{self.__class__.__name__ }(p={self.p})'


class Rescale:
    """
    Rescale the image in a sample to a given size.

    Args:
    output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
  """
    def __init__(self, output_size, interpolation = cv2.INTER_LINEAR ):
        assert isinstance(output_size, (int,tuple))
        self.output_size = output_size
        self.interpolation = interpolation

    def __call__(self, sample):
        clip = sample['clip']
        num_frames, h, w, num_channels = clip.shape
        if isinstance(self.output_size, int):
            if h > w:
                # set width to desired output size, rescale height to fit:
                new_h, new_w = self.output_size * h/w, self.output_size
            else:
                # other way around - set height, rescale width:
                new_h, new_w = self.output_size, self.output_size * w/h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        frames = np.zeros((num_frames, new_h, new_w, num_channels ))
        for i, frame in enumerate(clip):
            frame = cv2.resize(frame, dsize=(new_w, new_h),
                               interpolation=self.interpolation)
            if frame.ndim == 2:
                frame = frame[:, :, np.newaxis]
            frames[i, :, :, :] = frame
        clip = np.stack(frames)
        return {'clip': clip}
    def __repr__(self):
        return f'Rescale({self.output_size})'


class MatchHistograms:
    """ Match the histograms of all frames in the video to the first frame, to stabilize lighting flicker"""
    def __call__(self,sample):
        clip = sample["clip"]
        if torch.is_tensor(clip):
            clip = clip.cpu().numpy()
        frames = np.zeros_like(clip)
        ref = clip[0,...]
        frames[0,...] = ref
        multichannel = ref.shape[-1] == 3
        for i, frame in enumerate(clip[1:, ...]):
            matched = exposure.match_histograms(frame, ref, multichannel=multichannel)
            if frame.ndim == 2:
                matched = matched[:, :, np.newaxis]
            frames[i+1, :, :, :] = matched
        clip = np.stack(frames)
        return {'clip': clip}
    def __repr__(self):
        return 'MatchHistograms'


class ToTensor:
    """ Convert sample from numpy array to tensor"""
    def __call__(self, sample):
        clip = sample["clip"]
        assert isinstance(clip, np.ndarray), f'Expected numpy array got {type(clip)}'
        clip = np.transpose(clip, (3,0,1,2))
        # Assume proper representation for video in tensor is CXTxHxW
        # (color channels, time, height, width)
        clip = torch.from_numpy(clip)
        if not isinstance(clip, torch.FloatTensor):
          clip = clip.float()
        return {'clip': clip}
    def __repr__(self):
        return 'ToTensor'

