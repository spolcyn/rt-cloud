# Makes single-image slices from the available images

import nibabel as nib
import numpy as np
import os

IMAGE_FMT_STRING = "image-{}.nii"
DERIV_DIR = "deriv"

for i in range(1, 6):
    imagePath = IMAGE_FMT_STRING.format(i)
    print("Slicing", imagePath)
    image = nib.load(imagePath)

    # Convert to a 3-D NIfTI image, like what would result from converting a 3-D DICOM
    newData = image.dataobj[..., 0]
    print("New dimensions:", newData.shape)

    newImage = nib.Nifti1Image(newData, affine=image.affine, header=image.header)
    newImage.update_header()

    derivImagePath = os.path.join(DERIV_DIR, IMAGE_FMT_STRING.format(str(i) + "-deriv"))
    nib.save(newImage, derivImagePath)

print("All done")
