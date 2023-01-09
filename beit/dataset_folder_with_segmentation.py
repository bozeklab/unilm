from pathlib import Path

from torchvision.datasets.vision import VisionDataset

from PIL import Image

import os
import numpy as np
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from beit.dataset_folder import pil_loader, make_dataset, IMG_EXTENSIONS


def pil_pkl_loader(path: str) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
    extension = Path(path).suffix
    file_name = path.strip(extension)

    img = pil_loader(path)
    segmentation_pkl = np.load(f"{file_name}.pkl", allow_pickle=True)
    labels_pkl = np.load(f"{file_name}_cls.pkl", allow_pickle=True)
    return img, segmentation_pkl, labels_pkl


def default_loader(path: str) -> Any:
    return pil_pkl_loader(path)


class SegmentedDatasetFolder(VisionDataset):
    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(SegmentedDatasetFolder, self).__init__(root, transform=transform,
                                                     target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        while True:
            try:
                path, target = self.samples[index]
                sample = self.loader(path)
                break
            except Exception as e:
                print(e)
                index = random.randint(0, len(self.samples) - 1)

        if self.transform is not None:
            sample = [self.transform(sample[0]), sample[1]]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


class SegmentedImageFolder(SegmentedDatasetFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(SegmentedImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                                   transform=transform,
                                                   target_transform=target_transform,
                                                   is_valid_file=is_valid_file)
        self.imgs = self.samples
