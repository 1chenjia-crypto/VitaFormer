# CRC example images — source and licence

Two bright-field images of colorectal cancer organoids from the test set of the CRC organoid dataset
(Liu Y, et al. OrganoidViT: a vision transformer-based deep learning model for segmenting viable and nonviable organoids in bright-field images. Bio-Des Manuf. 2026;9:796–808. doi:10.1631/bdm.2500438).

| File | Selection rule (fixed before viewing) |
|---|---|
| `CRC_D12_J4M-CPT-11_WellE7_ChannelBF-2X_Seq0003_RGB_C1.png` | test image with the most annotated Dead pixels |
| `CRC_D6_CGC_WellD2_ChannelBF-2X_Seq0012_RGB_C1.png` | test image whose VitaFormer seed-42 per-image mIoU equals the median of the 167 test images |

The PNG files are lossless copies of the decoded original images (identical pixel values). The expected masks are the
VitaFormer seed-42 test-set masks reported in the paper (Figure 2D).

Licence: redistributed here with the permission of the data owners (the authors of the CRC dataset), for use as example images with VitaFormer. These images are not covered by the MIT License of this repository; contact the data owners for any other use.
