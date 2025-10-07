import os
import json
import copy
import sys
import importlib
import argparse
import random
import math
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from subprocess import DEVNULL, call
import numpy as np
import torch
import imageio
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

def _create_video_from_frames(render_dir, video_folder, video_text_file, start_frames, fps=24, num_frames=96):
    # get the sha256 from the directory name
    sha256 = os.path.basename(render_dir)

    # get all images in the directory
    images = os.listdir(render_dir)
    images = [os.path.join(render_dir, i) for i in images if i.endswith('.png')]
    # sort the images they are from 000.png to 049.png
    images = sorted(images, key=lambda x: int(os.path.basename(x).split('.')[0]))

    # check if the sha256 is in the metadata_df
    if len(images) >= num_frames:
        # get the video file name
        video_file = os.path.join(video_folder, f"{sha256}.mp4")

        # create the video from the images
        writer = imageio.get_writer(video_file, fps=fps)
        for image in images:
            img = imageio.imread(image)
            writer.append_data(img)
        writer.close()

        # write the video file name to the video.txt file
        with open(video_text_file, 'a') as f:
            f.write(f"videos/{sha256}.mp4\n")
        
        # write the start frame images[0] to the start_frames folder
        start_frame = os.path.join(start_frames, f"{sha256}.png")
        img = imageio.imread(images[0])
        imageio.imwrite(start_frame, img)

        return True
    else:
        return False


def _render_videos(file_path, sha256, output_dir):
    output_folder = os.path.join(output_dir, 'renders_videos', 'frames', sha256)
    video_folder = os.path.join(output_dir, 'renders_videos', 'videos')
    start_frames = os.path.join(output_dir, 'renders_videos', 'start_frames')
    video_text_file = os.path.join(output_dir, 'renders_videos', "video.txt")

    num_frames = 96
    radius_range_max = 0.3
    radius_range_min = 0.1
    current_radius = 1.5 + random.uniform(-radius_range_min, radius_range_max)
    fov=20
    yaws = torch.linspace(0, 2 * 3.1415, num_frames)
    yaws = yaws.tolist()
    base_pitch = 0.25 + random.uniform(-0.05, 0.05)
    pitch_sine_amplitude = 0.5 + random.uniform(-0.05, 0.05)
    pitch_sine_phase_shift = random.uniform(0, 2 * math.pi) 
    pitchs = [base_pitch + pitch_sine_amplitude * math.sin(y + pitch_sine_phase_shift) for y in yaws]
    radius = [current_radius] * len(yaws)
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
        video_created = _create_video_from_frames(output_folder, video_folder, video_text_file, start_frames, fps=24, num_frames=num_frames)
        if video_created:
            return {'sha256': sha256, 'video_rendered': True}



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

    os.makedirs(os.path.join(opt.output_dir, 'renders_videos'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'renders_videos', 'frames'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'renders_videos', 'videos'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'renders_videos', 'start_frames'), exist_ok=True)
    
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
        if os.path.exists(os.path.join(opt.output_dir, 'renders_videos', sha256, 'transforms.json')):
            records.append({'sha256': sha256, 'eval_rendered': True})
            metadata = metadata[metadata['sha256'] != sha256]
    
    print(f'Processing {len(metadata)} objects...')

    # process objects
    func = partial(_render_videos, output_dir=opt.output_dir)
    video_rendered = dataset_utils.foreach_instance(metadata, opt.output_dir, func, max_workers=opt.max_workers, desc='Rendering objects')
    video_rendered = pd.concat([video_rendered, pd.DataFrame.from_records(records)])
    video_rendered.to_csv(os.path.join(opt.output_dir, f'video_rendered_{opt.rank}.csv'), index=False)
