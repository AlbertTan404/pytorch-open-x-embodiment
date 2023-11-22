# Data pre-processing code on Open-X-Embodiment for PyTorch users. 


### Usages:
With the repo, you can
- Convert tfds to h5 file with _convert_tfds_to_h5.py_  # converting large datasets takes MASSIVE disk space (up to 8 TB for kuka)

- Visualize the datasets with the processed h5 file with _check_data.ipynb

- Extract raw images with _move_h5_image_to_png.py_

- Extract image and language features for most efficient policy model training with _extract_language_features.py_ and _extract_image_features.py_ (we use R3M and CLIP, and it easy for you to customize it)

- Normalize actions according to the statistics for unified training with _normalize_actoins.py_ and _rt-x_data_cfg.yaml_  # 

---

### Features:
- For extracting features, we use multi-processing among datasets for better efficiency
- For converting tfds to hdf5 files, we also support parallel processing by setting the --index option  # default turned off