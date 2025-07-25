import os
from pathlib import Path
import shutil
from dotenv import load_dotenv

load_dotenv()
data_dir = Path(os.getenv("BrainMetShare"))
out_dir = Path(os.getenv("BrainMetShare_nnUNET"))

imagesTr_dir = out_dir / "imagesTr"
labelsTr_dir = out_dir / "labelsTr"

imagesTr_dir.mkdir(parents=True, exist_ok=True)
labelsTr_dir.mkdir(parents=True, exist_ok=True)

for i, mri_dir in enumerate(data_dir.iterdir()):
    if not mri_dir.is_dir():
        continue

    t1_post = (mri_dir / "t1_gd.nii.gz")
    mask = (mri_dir / "seg.nii.gz")

    case_id = f"patient{i:03d}"

    # nnU-Net expects _0000 suffix for the first modality
    out_img_path = imagesTr_dir / f"{case_id}_0000.nii.gz"
    out_mask_path = labelsTr_dir / f"{case_id}.nii.gz"

    shutil.copy(t1_post, out_img_path)
    shutil.copy(mask, out_mask_path)

    print(f"Copied {mri_dir.name} to {case_id}")