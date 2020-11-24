# Transforms, adapted from the pytorch documentation:
# https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html
import cv2
import numpy as np
import torch


class Rescale:
    """Rescale the image in a sample to a given size.

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
        frames = np.zeros((num_frames, new_w, new_h, num_channels ))
        for i, frame in enumerate(clip):
            frame = cv2.resize(frame, dsize=(new_w, new_h),
                               interpolation=self.interpolation)
            if frame.ndim == 2:
                frame = frame[:, :, np.newaxis]
            frames[i, :, :, :] = frame
        clip = np.stack(frames)
        return {'clip': clip}


class ToTensor:
    """ Convert sample from numpy array to tensor"""
    def __call__(self, sample):
        clip = sample["clip"]
        assert isinstance(clip, np.ndarray), f'Expected numpy array got {type(clip)}'
        clip = np.transpose(clip, (3,0,1,2))
        # Assume proper representation for video in tensor is TxHxWxC
        # (time, height, width, color channels)
        clip = torch.from_numpy(clip)
        if not isinstance(clip, torch.FloatTensor):
          clip = clip.float()
        return {'clip': clip}

