import os
import json
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from subprocess import DEVNULL, call
import numpy as np
import torch
from utils import sphere_hammersley_sequence, generate_views_from_angles


BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/home/damian/Projects/Diffus3D/'
# BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'
BLENDER_PATH = "/home/damian/Projects/Diffus3D/blender-3.2.2-linux-x64/blender"

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')

def _render_eval(file_path, sha256, output_dir):
    output_folder = os.path.join(output_dir, 'renders_eval', sha256)
    
    # render_90_and_70
    # azimuths =   [0,  30,  60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    # elevations = [90, 100, 70, 90, 100,  70,  90, 100,  70,  90, 100,  70]
    # render_90
    # azimuths =    [ 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    # elevations =  [90, 90, 90, 90,  90,  90,  90,  90,  90,  90,  90,  90]
    # render_70
    # azimuths =    [ 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    # elevations =  [70, 70, 70, 70,  70,  70,  70,  70,  70,  70,  70,  70]
    # render video
    num_frames = 96
    # azimuths =  torch.linspace(0, 360, num_frames + 1)
    # azimuths = azimuths.tolist()[:-1]
    # elevations =  [70] * len(azimuths)
    # views = generate_views_from_angles(azimuths, elevations, fixed_radius=2.0)  
    r=1.5
    fov=20
    yaws = torch.linspace(0, 2 * 3.1415, num_frames)
    yaws = yaws.tolist()
    pitchs = [0.5] * len(yaws)
    radius = [r] * len(yaws)
    fov = [fov] * len(yaws)
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} 
             for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    
    args = [
        BLENDER_PATH, '-b', '-P', os.path.join('/mnt/damian/Projects/TRELLIS/dataset_toolkits', 'blender_script', 'render.py'),
        '--',
        '--views', json.dumps(views),
        '--object', os.path.expanduser(file_path),
        '--output_folder', os.path.expanduser(output_folder),
        '--resolution', '480',
        # '--save_glb_mesh',
    ]
    if file_path.endswith('.blend'):
        args.insert(1, file_path)
    
    call(args, stdout=DEVNULL)
    
    if os.path.exists(os.path.join(output_folder, 'transforms.json')):
        return {'sha256': sha256, 'cond_rendered': True}


if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=8)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    os.makedirs(os.path.join(opt.output_dir, 'renders_eval'), exist_ok=True)
    
    # install blender
    print('Checking blender...', flush=True)
    _install_blender()

    # get file list
    if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    if opt.instances is None:
        metadata = metadata[metadata['local_path'].notna()]
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'eval_rendered' in metadata.columns:
            metadata = metadata[metadata['eval_rendered'] == False]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    # filter out objects that are already processed
    for sha256 in copy.copy(metadata['sha256'].values):
        if os.path.exists(os.path.join(opt.output_dir, 'renders_eval', sha256, 'transforms.json')):
            records.append({'sha256': sha256, 'eval_rendered': True})
            metadata = metadata[metadata['sha256'] != sha256]
    
    print(f'Processing {len(metadata)} objects...')

    # process objects
    func = partial(_render_eval, output_dir=opt.output_dir)
    cond_rendered = dataset_utils.foreach_instance(metadata, opt.output_dir, func, max_workers=opt.max_workers, desc='Rendering objects')
    cond_rendered = pd.concat([cond_rendered, pd.DataFrame.from_records(records)])
    cond_rendered.to_csv(os.path.join(opt.output_dir, f'eval_rendered_{opt.rank}.csv'), index=False)
