# Custom Dataset, adapted from the following sources:
# https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html
# https://github.com/MohsenFayyaz89/PyTorch_Video_Dataset/blob/master/GeneralVideoDataset.py
# https://github.com/hassony2/torch_videovision/blob/master/torchvideotransforms/volume_transforms.py
import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from skimage import exposure


class VideoDataset(Dataset):
    def __init__(self, directory, num_frames, color_channels=1, transform=None,
                 div_255=True, match_hists=False):
        self.dir = directory
        self.transform = transform
        self.num_frames = num_frames
        self.color_channels = color_channels
        self.div_255 = div_255
        self.file_paths = []
        self.match_hists = match_hists
        self.load_file_names()

    def load_file_names(self):
        for root, directories, files in os.walk(self.dir):
            for filename in files:
                self.register_filename(root, filename)

    def register_filename(self, root, filename):
        if filename.endswith('.avi'):
            filepath = os.path.join(root, filename)  # Assemble the full path
            self.file_paths.append(filepath)  # Add it to the list

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx=idx.tolist()
        # get a tensor representing the video, dimensions will be TxHxWxC
        cap = cv2.VideoCapture(self.file_paths[idx])
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = np.zeros((self.num_frames, height, width, self.color_channels))
        for i in range(self.num_frames):
            ret, frame = cap.read()
            if ret:
                # Convert from BGR to either Grayscale or RGB if input is an image:
                # optic flow will remain as is with two "color" channels
                if self.color_channels == 1:
                    frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = frame[:, :, np.newaxis]
                elif self.color_channels == 3:
                    frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame = frame[:,:,:-1] # in optic flow, the last channel is artificially added
                if self.div_255:
                    frame = frame/255
                if self.match_hists:
                    if i==0:
                        ref = frame.copy()
                    else:
                        frame = exposure.match_histograms(frame, ref, multichannel=(self.color_channels>1))
                frames[i, :, :, :] = frame
        frames = np.stack(frames)
        cap.release()
        cv2.destroyAllWindows()
        # check if the video has the required amount of frames:
        assert frames.shape[0] == self.num_frames
        sample = {'clip': frames}
        if self.transform:
            sample = self.transform(sample)
        return sample
