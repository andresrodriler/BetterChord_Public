import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


class ChordCNN(nn.Module):

    # chromatic_notes: Number of notes to predict (12 for each note in the chromatic scale)
    def __init__(self, chromatic_notes=12):

        super(ChordCNN, self).__init__()

        # Convolutional layer 1 to extract features from the spectrogram (with batch normalization)
        # Shared
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        # Convolutional layer 2 to extract features from the spectrogram (with batch normalization)
        # Shared
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        # CQT Spectrogram Input Is (84, 119)
        # After the two maxpooling/conv layers in order will be:
            # Height: 84 → 42 → 21
            # Width: 119 → 59 → 29
            # Channels: 1 to 32 to 64
        # So input to first FC layer will be 64 × 21 × 29 = 38976
        self.flat_size = 64 * 21 * 29  # 38976

        # Dropout layer
        self.dropout = nn.Dropout(0.4)

        # This point on, fully connected layers split into three "paths"
            # 1) For detecting which 12 notes are present within 12 vector array
            # 2) For detecting which of the 12 notes is the root
            # 3) For detecting the bass note

        # Note Path (mult-label), using BCEWithLogitsLoss during training
        self.note_fc1 = nn.Linear(self.flat_size, 256)
        self.note_fc2 = nn.Linear(256, 128)
        self.note_fc3 = nn.Linear(128, chromatic_notes)

        # Root Path (single-label), using CrossEntropyLoss during training
        self.root_fc1 = nn.Linear(self.flat_size, 256)
        self.root_fc2 = nn.Linear(256, 128)
        self.root_fc3 = nn.Linear(128, chromatic_notes)

        # Bass Path (single-label), using CrossEntropyLoss during training
        self.bass_fc1 = nn.Linear(self.flat_size, 256)
        self.bass_fc2 = nn.Linear(256, 128)
        self.bass_fc3 = nn.Linear(128, chromatic_notes)


    def forward(self, x):
        # Pass through conv layers (Shared for root and chromatic scale vector)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Second conv layer (Shared for root and chromatic scale vector)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Flatten
        x = x.view(x.size(0), -1)

        # This point on, fully connected layers split into three "paths"
            # 1) For detecting which 12 notes are present within 12 vector array
            # 2) For detecting which of the 12 notes is the root
            # 3) For detecting the bass note

        # Note Path (mult-label), dropout for it and input from final conv layer 2
        note_x   = self.dropout(F.relu(self.note_fc1(x)))
        note_x   = self.dropout(F.relu(self.note_fc2(note_x)))
        note_out = self.note_fc3(note_x)  # raw logits → BCEWithLogitsLoss

        # Root Path (single-label), dropout for it and input from final conv layer 2
        root_x   = self.dropout(F.relu(self.root_fc1(x)))
        root_x   = self.dropout(F.relu(self.root_fc2(root_x)))
        root_out = self.root_fc3(root_x)  # raw logits → CrossEntropyLoss

        # Bass Path (single-label), dropout for it and input from final conv layer 2
        bass_x   = self.dropout(F.relu(self.bass_fc1(x)))
        bass_x   = self.dropout(F.relu(self.bass_fc2(bass_x)))
        bass_out = self.bass_fc3(bass_x)  # raw logits → CrossEntropyLoss

        return note_out, root_out, bass_out