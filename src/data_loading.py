import random
import numpy as np
import torch
import tqdm as tqdm
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

sequence_length, overlap_length, batch_size = 60, 25, 24


rand = random.Random(42)


class EventDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame = self.data[idx]
        target = self.targets[idx]

        return self.transform(frame, target)

    def transform(self, frames, targets):

        rand_region = rand.randint(0, 8)

        if rand_region == 0:
            cropped_frames = TF.center_crop(frames, (200, 200))
            cropped_targets = TF.resize(TF.center_crop(targets, (50, 50)), (64, 64))

        elif rand_region == 1:
            cropped_frames = TF.crop(frames, 0, 0, 200, 200)
            cropped_targets = TF.resize(TF.crop(targets, 0, 0, 50, 50), (64, 64))

        elif rand_region == 2:
            cropped_frames = TF.crop(frames, 0, 256 - 200, 200, 200)
            cropped_targets = TF.resize(TF.crop(targets, 0, 64 - 50, 50, 50), (64, 64))

        elif rand_region == 3:
            cropped_frames = TF.crop(frames, 256 - 200, 256 - 200, 200, 200)
            cropped_targets = TF.resize(TF.crop(targets, 64 - 50, 64 - 50, 50, 50), (64, 64))

        elif rand_region == 4:
            cropped_frames = TF.crop(frames, 256 - 200, 0, 200, 200)
            cropped_targets = TF.resize(TF.crop(targets, 64 - 50, 0, 50, 50), (64, 64))

        elif rand_region == 5:
            cropped_frames = TF.crop(frames, 0, 28, 200, 200)
            cropped_targets = TF.resize(TF.crop(targets, 0, 7, 50, 50), (64, 64))

        elif rand_region == 6:
            cropped_frames = TF.crop(frames, 28, 256 - 200, 200, 200)
            cropped_targets = TF.resize(TF.crop(targets, 7, 64 - 50, 50, 50), (64, 64))

        elif rand_region == 7:
            cropped_frames = TF.crop(frames, 256 - 200, 28, 200, 200)
            cropped_targets = TF.resize(TF.crop(targets, 64 - 50, 7, 50, 50), (64, 64))

        elif rand_region == 8:
            cropped_frames = TF.crop(frames, 28, 0, 200, 200)
            cropped_targets = TF.resize(TF.crop(targets, 7, 0, 50, 50), (64, 64))
        return cropped_frames, cropped_targets


def create_frame_sequences(data, sequence_length, overlap_length, batch_size):
    sequences = []
    start_index = 0

    while start_index + sequence_length <= len(data):
        sequence = data[start_index : start_index + sequence_length]
        sequences.append(sequence)
        start_index += sequence_length - overlap_length
    num_sequences = (len(sequences) // batch_size) * batch_size
    sequences = sequences[:num_sequences]

    return sequences


def create_target_sequences(data, sequence_length, overlap_length, batch_size):
    sequences = []
    start_index = 0

    while start_index + sequence_length <= len(data[0]):
        sequence1 = data[0][start_index : start_index + sequence_length]
        sequence2 = data[1][start_index : start_index + sequence_length]
        sequence3 = data[2][start_index : start_index + sequence_length]
        sequence4 = data[3][start_index : start_index + sequence_length]

        sequences.append([sequence1, sequence2, sequence3, sequence4])
        start_index += sequence_length - overlap_length

    num_sequences = (len(sequences) // batch_size) * batch_size
    sequences = sequences[:num_sequences]

    return torch.from_numpy(np.asarray(sequences))


def get_data(fpaths):
    frames_sequences = []
    target_sequences = []

    for fpath in fpaths:
        data = torch.load(fpath)
        if not data[0].is_coalesced():
            data[0] = data[0].coalesce()

        for i in range(4):
            if not data[i + 1].is_coalesced():
                data[i + 1] = data[i + 1].coalesce()

        frames_tensor = data[0].to_dense().float()
        targets = []
        for i in range(4):
            targets.append(data[i + 1].to_dense().float())

        temp1 = create_frame_sequences(frames_tensor, sequence_length, overlap_length, batch_size)

        frames_sequences.extend(temp1)

        temp2 = create_target_sequences(targets, sequence_length, overlap_length, batch_size)

        target_sequences.extend(temp2)

    # kfold = KFold(n_splits=3, shuffle=True, random_state=42)

    # for train_index, test_index in kfold.split(frames_sequence):

    #     X_train, X_test = frames_sequence[train_index], frames_sequence[test_index]
    #     y_train, y_test = target_sequences[train_index], target_sequences[test_index]

    X_train, X_test, y_train, y_test = train_test_split(
        frames_sequences,
        target_sequences,
        test_size=0.2,
        random_state=42,
    )

    dataset = EventDataset(X_train, y_train)
    test_dataset = EventDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=False,
        prefetch_factor=8,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=2,
        pin_memory=False,
        prefetch_factor=8,
    )

    return train_loader, test_loader
