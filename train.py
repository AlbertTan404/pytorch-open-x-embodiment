import argparse
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.trainer import Trainer

import math
import importlib
from datetime import datetime
from omegaconf.dictconfig import DictConfig


def get_timestamp():
    return datetime.now().strftime('%Y%m%d-%H%M%S')


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config, extra_kwargs=dict()):
    config_dict = dict(config)
    if not "target" in config_dict:
        if config_dict == '__is_first_stage__':
            return None
        elif config_dict == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    target_kwargs = dict(config_dict.get('kwargs', dict()))

    for k, v in target_kwargs.items():
        if isinstance(v, DictConfig) and 'target' in v.keys():
            target_kwargs[k] = instantiate_from_config(v)
    target_kwargs.update(extra_kwargs)
    return get_obj_from_str(config_dict["target"])(**target_kwargs)


def get_train_val_loader(dataset, **dataloader_kwargs):
    train_ds, val_ds = dataset.split_train_val(train_ratio=0.95)
    train_loader = DataLoader(dataset=train_ds, **dataloader_kwargs, shuffle=True)
    val_loader = DataLoader(dataset=val_ds, **dataloader_kwargs, shuffle=False)
    return train_loader, val_loader


def preprocess_config(config, args):
    # set timestamp
    task = args.task
    project_name = config.model.target.split('.')[-2] + '_logs'
    config.trainer.kwargs.logger.kwargs.project = project_name
    config.trainer.kwargs.logger.kwargs.name = f'{get_timestamp()}-{task}'

    # overriding horizon
    config.horizon = args.horizon
    config.model.kwargs.model_kwargs.horizon = args.horizon
    config.dataset.kwargs.horizon = args.horizon

    # devices
    devices = args.devices
    if devices is not None:
        devices = devices.split(',')
        devices = [int(rank) for rank in devices]
        config.trainer.kwargs.devices = devices

    # avoid gpu rank overflow
    device_count = torch.cuda.device_count()
    if len(config.trainer.kwargs.devices) > device_count:
        config.trainer.kwargs.devices = list(range(device_count))
        print(f'using {device_count} devices')

    # batch size for ddp
    total_bs = config.dataloader.batch_size
    num_devices = len(config.trainer.kwargs.devices)
    bs_per_device = total_bs // num_devices
    real_bs = bs_per_device * num_devices
    if real_bs != total_bs:
        print(f'real batch size is {real_bs}')
    config.dataloader.batch_size = bs_per_device

    # dataset/tasks/mode
    data_cfg = OmegaConf.load(f'{task}_data_cfg.yaml')

    datasets_cfg = data_cfg.datasets
    config.dataset.kwargs.root_dir = f'YOUR_DATASET_ROOT_DIR_HERE/{task}_processed'
    config.dataset.kwargs.data_cfg = datasets_cfg
    config.dataset.kwargs.dataset_names = [key for key in datasets_cfg.keys() if key[0] != '_']
    config.dataset.kwargs.average_step_per_episode = data_cfg.average_step_per_episode

    # feature dimension:
    if config.dataset.kwargs.feature_type[:3] == 'r3m':
        config.model.kwargs.model_kwargs.language_feature_dim = 768
    else:  # clip
        config.model.kwargs.model_kwargs.language_feature_dim = 512

    return config


def get_parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config_name',
        default='example_cfg'
    )
    parser.add_argument(
        '--task',
        default='rt-x'
    )
    parser.add_argument(
        '--devices',
        type=str,
        default='0',
    )
    parser.add_argument(
        '--horizon',
        type=int,
        default=16
    )

    return parser.parse_args()


def main():
    args = get_parser_args()

    raw_config = OmegaConf.load(f'{args.config_name}.yaml')
    OmegaConf.resolve(raw_config)
    config = preprocess_config(raw_config, args)

    pl.seed_everything(config.seed)

    model: pl.LightningModule = instantiate_from_config(config.model, extra_kwargs={"all_config": config})

    dataset = instantiate_from_config(config.dataset)
    train_loader, val_loader = get_train_val_loader(dataset=dataset, **config.dataloader)

    epoch_length = len(train_loader) // len(config.trainer.kwargs.devices)
    config.model.kwargs.training_kwargs['num_training_steps'] = epoch_length * config.trainer.kwargs.max_epochs

    trainer: Trainer = instantiate_from_config(config.trainer)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()
