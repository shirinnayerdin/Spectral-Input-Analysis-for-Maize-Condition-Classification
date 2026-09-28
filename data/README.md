# Dataset

This project uses a representative mini dataset derived from the maize UAV dataset described in the source study.

## Dataset Structure

Two aligned versions of the mini dataset are used:

- `mini_dataset_3ch`: 3-channel RGB images
- `mini_dataset_6ch`: 6-channel images

The dataset contains five classes:

- `bare_soil`
- `healthy`
- `high_stress`
- `low_stress`
- `rust`

Each class contains 150 images, resulting in a total of 750 images.

## Sampling

The mini dataset was created by random sampling with a fixed seed of 42.

The original number of available images before sampling was:

| Class | Available Images | Selected Images |
|---|---:|---:|
| low_stress | 1066 | 150 |
| high_stress | 1252 | 150 |
| rust | 1303 | 150 |
| healthy | 728 | 150 |
| bare_soil | 1510 | 150 |

Images were pooled from the original training and test folders before the 150-image-per-class subset was sampled.

Therefore, the train/validation/test split used in this repository should be interpreted as an exploratory patch-level split of the prepared mini dataset rather than an independent split preserving the original source partitions.

## Dataset Setup

To run the notebook, place the prepared datasets inside the `data/` directory as follows:

```text
data/
├── mini_dataset_3ch/
│ ├── bare_soil/
│ ├── healthy/
│ ├── high_stress/
│ ├── low_stress/
│ └── rust/
│
└── mini_dataset_6ch/
    ├── bare_soil/
    ├── healthy/
    ├── high_stress/
    ├── low_stress/
    └── rust/


The notebook expects the dataset path:

`../data`

when executed from the `notebooks/` directory.

## Data Availability

The mini dataset used for the reported experiments is not included directly in this repository.

For information about the source maize UAV dataset and its data-generation methodology, see the associated study:

Ç. Suiçmez, C. Yılmaz, H. T. Kahraman, and M. A. Erdoğan,  
“UAV-Derived Multispectral Datasets and Index-Guided Segmentation for Maize Water Stress and Common Rust Detection Under Real Field Conditions,”  
*Applied Sciences*, 2026.

DOI: https://doi.org/10.3390/app16146860

## Important Note

The experiments in this repository were conducted on the 750-image prepared mini dataset described above. Results should therefore not be interpreted as independent field-level or cross-site generalization performance.
