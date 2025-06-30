from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from dotenv import load_dotenv
import os
from torchvision.transforms.functional import resize as TF_resize

load_dotenv()
data_dir = Path(os.getenv("BrainMetShare"))

def largest_slice(mask):
    slice_sums = mask.sum(axis=(0, 1))
    z = int(np.argmax(slice_sums))
    return z

# save all mri images as npy files
data_dir = Path(data_dir)
out_dir = data_dir.parent.parent.parent / 'BrainMetShare_t1_gd'
out_dir.mkdir(parents=True, exist_ok=True)

for mri_dir in data_dir.iterdir():
    if not mri_dir.is_dir():
        continue

    t1_post = nib.load(mri_dir / "t1_gd.nii.gz").get_fdata()
    mask = nib.load(mri_dir / "seg.nii.gz").get_fdata()

    # Normalize
    def normalize(x):
        return (x - x.mean()) / (x.std() + 1e-8)

    t1_post = normalize(t1_post)

    # Convert to tensors and resize
    target_size = (256, 256)
    t1_tensor = TF_resize(torch.tensor(t1_post).unsqueeze(0), target_size).squeeze(0)
    mask_tensor = TF_resize(torch.tensor(mask).unsqueeze(0), target_size).squeeze(0)

    # Save
    new_dir = out_dir / mri_dir.name
    new_dir.mkdir(parents=True, exist_ok=True)
    np.save(new_dir / "image.npy", t1_tensor.numpy().astype(np.float32))
    np.save(new_dir / "mask.npy", mask_tensor.numpy().astype(np.uint8))

    print(f"Saved: {new_dir / 'image.npy'}, {new_dir / 'mask.npy'}")