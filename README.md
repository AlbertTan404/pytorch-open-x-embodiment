# Data pre-processing and training code on Open-X-Embodiment for PyTorch users. 


### Dataset access:
Refer to the [rt-x official repo](https://github.com/google-deepmind/open_x_embodiment#dataset-access)

### Usages:
With this repo, you could:
- Convert tfds to h5 file with *convert_tfds_to_h5.py*  # converting large datasets takes MASSIVE disk space. (up to 8 TB for kuka)

- Visualize the processed h5 file with *check_data.ipynb*.

- Extract raw images with *extract_images.py*.

- Extract image and language features *extract_language_features.py* and *extract_image_features.py*. (we use R3M and CLIP, and it's easy to customize it)

- Normalize actions according to the statistics with *normalize_actoins.py* and *rt-x_data_cfg.yaml*. 

and

- Customize you model and directly train it!


### Features:
- For extracting features, we use multi-processing among datasets for better efficiency.
- For converting tfds to hdf5 files, we also support parallel processing by setting the --index argument.  # turned off by default


### Environment
In your python environment:

- install tf and tfds
```
pip install tensorflow tensorflow-datasets
```

- some basic libraries
```
conda install h5py yaml jupyter tqdm omegaconf gdown matplotlib
```

- install pytorch (version not restricted)
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

- optional:
```
conda install lightning transformers diffusers  # for model training
```

```
pip install git+https://github.com/openai/CLIP.git
```

```
pip install git+https://github.com/facebookresearch/r3m.git
```
