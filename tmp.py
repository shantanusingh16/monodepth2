import os
import shutil

with open('splits/scannet_full/eval_files.txt', 'r') as f:
    filepaths = f.readlines()

data_dir = ''
output_dir = ''

for filepath in filepaths:
    folder, fileidx = filepath.split()

    for dir in ['imgs', 'depths', 'masks']:
        src_rgb_path = os.path.join(data_dir, dir, folder, '{}.png'.format(fileidx))
        dst_rgb_path = os.path.join(output_dir, dir, folder, '{}.png'.format(fileidx))

        os.makedirs(dst_rgb_path, exist_ok=True)

        shutil.copy(src_rgb_path, dst_rgb_path)

