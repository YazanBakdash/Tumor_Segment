from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from dotenv import load_dotenv
import os
import torchvision.transforms.functional as TF

load_dotenv()
data_dir = Path(os.getenv("DATA_DIR"))

def load_image_and_mask(dir):
    image_path = next(dir.glob("*image*"))
    image = nib.load(image_path).get_fdata()
    mask_paths = sorted(dir.glob("*mask*"))
    combined_mask = np.zeros_like(image)
    for path in mask_paths:
        mask = nib.load(path).get_fdata()
        combined_mask = np.logical_or(combined_mask, mask)
    return image, combined_mask.astype(np.float32)

def largest_slice(mask):
    slice_sums = mask.sum(axis=(0, 1))
    z = int(np.argmax(slice_sums))
    return z

# save all mri images as npy files
data_dir = Path(data_dir)
out_dir = data_dir.parent / 'BCBM_npy'
out_dir.mkdir(parents=True, exist_ok=True)

for mri_dir in data_dir.iterdir():
    if not mri_dir.is_dir():
        continue

    image, mask = load_image_and_mask(mri_dir)
    z = largest_slice(mask)

    target_size = (256, 256)

    image_slice = torch.tensor(image[:, :, z], dtype=torch.float32).unsqueeze(0)
    image_slice = TF.resize(image_slice, target_size)

    mask_slice = torch.tensor(mask[:, :, z], dtype=torch.float32).unsqueeze(0)
    mask_slice = TF.resize(mask_slice, target_size)

    # Save as .npy (optional: squeeze to [H, W])
    new_dir = out_dir / mri_dir.name
    new_dir.mkdir(parents=True, exist_ok=True)

    np.save(new_dir / "image.npy", image_slice.numpy())
    np.save(new_dir / "mask.npy", mask_slice.numpy().astype(np.uint8))

    print(f"Saved: {new_dir / 'image.npy'}, {new_dir / 'mask.npy'}")