import os
import random

import click
from tqdm import tqdm

from parse_aml import aml_to_dict, aml_dict_to_data_dict

#val_dir = 'D:/data/objdet_val'
#test_fraction = 0.1
#txt_dir = 'ground_truths'

@click.command()
@click.argument('val_dir')
@click.argument('txt_dir')
@click.argument('test_fraction', type=float)
def prepare_test_imgs(val_dir: str, txt_dir: str, test_fraction: float):

    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)

    test_imgs_paths, test_amls_paths = [], []

    for val_source in ['ive', 'iv_cloud', 'mscoco', 'voc']:
        all_files = os.listdir(os.path.join(val_dir, val_source, 'imgs'))
        imgs = list(filter(lambda x: '.aml' not in x, all_files))
        amls = list(filter(lambda x: '.aml' in x, all_files))

        test_ix = random.sample(list(range(len(imgs))), int(test_fraction*len(imgs)))
        test_imgs_paths += [os.path.join(val_dir, val_source, 'imgs', imgs[i]) for i in test_ix]
        test_amls_paths += [os.path.join(val_dir, val_source, 'imgs', amls[i]) for i in test_ix]

    for aml_path in tqdm(test_amls_paths):
        aml_dict = aml_to_dict(aml_path)
        data_dict = aml_dict_to_data_dict(aml_path, aml_dict)
        obj_descr = []
        txt_path = os.path.join(txt_dir, os.path.basename(aml_path).replace('aml', 'txt'))
        for obj_dict in data_dict['object']:
            label = obj_dict['name']
            left = obj_dict['bndbox']['xmin']
            right = obj_dict['bndbox']['xmax']
            top = obj_dict['bndbox']['ymin']
            bottom = obj_dict['bndbox']['ymax']
            obj_descr.append(f"{label} {left} {top} {right} {bottom} \n")

        with open(txt_path, 'w') as f:
            f.writelines(obj_descr)


if __name__ == '__main__':
    prepare_test_imgs()
