r"""Convert the IV dataset to TFRecord file format.

Example usage:
    python ivlog2tfrecord.py \
        --label_map_path=./label_maps/mscoco_label_map.pbtxt \
        --data_dir=agilence_vid20110619 \
        --output_path=/path/to/out.tfrecord

The default values for 2 parameters are:
    --label_map_path = .../tensorflow/models/research/object_detection/data/mscoco_label_map.pbtxt
    --output_path = ./<data_dir>.tfrecord
    
This script is based on:
    - the scripts 'iv2pascal_agilence.py' and 'create_pascal_tf_record.py' by Denis Pimankin
    - the script 'create_iv_detection_record.py' by Alexey Vihirev
"""

import hashlib
import io
import logging
import sys, os, os.path
import random

from collections import defaultdict, OrderedDict
#from lxml import etree
import xmltodict
import q
from PIL import Image
import numpy as np
import tensorflow as tf

## dmburd:
#sys.path.append(os.getcwd())
##

#--------
# add 'object_detection' location to sys.path in an adaptive way:
site_pkgs = filter(lambda p: p.endswith("site-packages"), sys.path)
site_pkgs = list(site_pkgs)
#print('site_pkgs:', site_pkgs)
obj_det_dir_saved = os.getcwd()
for d in site_pkgs:
    rsrch = os.path.join(d, "tensorflow", "models", "research",)
    sys.path.insert(0, rsrch)
    obj_det_dir = os.path.join(rsrch, "object_detection",)
    obj_det_dir_saved = obj_det_dir
    sys.path.insert(0, obj_det_dir)
#--------

from utils import dataset_util
from utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to the dataset')
flags.DEFINE_string('output_path', '', 'Path to the output TFRecord file')
flags.DEFINE_string('label_map_path', os.path.join(obj_det_dir_saved, 'data', 'mscoco_label_map.pbtxt'), 'Path to label map proto')
FLAGS = flags.FLAGS
    
LABELS_IV_TO_MSCOCO = {
    'eOC_Human':    '/m/01g317',#person, id = 1 in mscoco
    'eOC_Vehicle':  '/m/0k4j',  #car, id = 3
    'eOC_Train':    '/m/07jdr', #train, id = 7
    'eOC_Pet':      '/m/01yrx', #cat, id = 17
    'eOC_Airplane': '/m/05czz6l', #airplane, id = 5
    'eOC_Other':    '/m/0kmg4', #teddy bear, id = 88
}


def aml_to_dict(aml_path):
    """
    One of the simplest ways to parse an xml file
    is to use xmltodict module.
    
    Args:
        aml_path: The path to the .aml file (xml-like ivlog file format)
    
    Returns:
        The contents of the .aml file as a dictionary (OrderedDict)
    """
    with tf.gfile.GFile(aml_path, 'r') as f:
        markup = f.read()
    
    di = xmltodict.parse(markup)
    #print(di)
    #b = xmltodict.unparse(di, pretty=True, indent='  ')
    #print(b)
    return di
    
    
def aml_dict_to_data_dict(aml_path, aml_dict):
    """
    Fill another dictionary using the data from the dictionary 
    returned by aml_to_dict().
    The result is close in structure (fields) to the data 
    from a PASCAL VOC description file (also xml-like).
    
    Args:
        aml_path: The path to the .aml file (xml-like ivlog file format)
        aml_dict: The dictionary returned by aml_to_dict() for aml_path file
        
    Returns:
        OrderedDict with the data required to fill a tf.train.Example
        
    The image file corresponding to the aml file is obtained by the following:
        1) refer to 'DataSource' tag in the aml file
        2) if the previous attempt fails ('DataSource' tag is absent),
           search for the files with the same name as the aml file
           and with extensions from the list ['.png', '.jpg', 'jpeg']
           at the same directory where the aml file is located
        3) if both previous attempts fail, return None
    """
    
    #print("  -> ", aml_path)
    data = OrderedDict.fromkeys(['filename', 'size', 'object'])
    if 'DataSource' in aml_dict['IvLog']['Parameters']:
        #print("  -> DataSource tag is found")
        data['filename'] = aml_dict['IvLog']['Parameters']['DataSource']
    else:
        #print("  -> DataSource tag is NOT FOUND")
        (dirname, basename) = os.path.split(aml_path)
        (namenoext, ext) = os.path.splitext(basename)
        for ext in ['.png', '.jpg', 'jpeg']:
            attempt = os.path.join(dirname, namenoext + ext)
            if os.path.isfile(attempt):
                #print("  -> '%s' file found" % ext)
                data['filename'] = attempt
    
    if data['filename'] == None:
        # the image is not found
        return None
    
    data['size'] = OrderedDict.fromkeys(['width', 'height'])
    data['size']['width'] = aml_dict['IvLog']['Parameters']['FrameWidth']
    data['size']['height'] = aml_dict['IvLog']['Parameters']['FrameHeight']
    data['object'] = []
    
    if 'Objects' in aml_dict['IvLog']['Frames']['Frame']:
        li = aml_dict['IvLog']['Frames']['Frame']['Objects']['Object']
        if type(li) not in [list, tuple]:
            li = [li,]
        for objdi in li:
            obj = OrderedDict.fromkeys(['difficult', 'name', 'bndbox', 'truncated', 'pose'])
            obj['difficult'] = 0
            obj['truncated'] = 0
            obj['pose'] = "Unspecified"
            obj['name'] = objdi['@Class']
            obj['bndbox'] = OrderedDict.fromkeys(['xmin', 'ymin', 'xmax', 'ymax'])
            obj['bndbox']['xmin'] = int(objdi['@X'])
            obj['bndbox']['ymin'] = int(objdi['@Y'])
            obj['bndbox']['xmax'] = int(objdi['@X']) + int(objdi['@Width'])
            obj['bndbox']['ymax'] = int(objdi['@Y']) + int(objdi['@Height'])
            data['object'].append(obj)
    
    return data


def create_mask_simple_rectangle(bbox_w, bbox_h, rel_margins, trans_len_rel):
    # bbox_w, bbox_h = true width and height of the original bbox (without margins)
    w = int(round(bbox_w * (1.0 + rel_margins[0] + rel_margins[2])))
    h = int(round(bbox_h * (1.0 + rel_margins[1] + rel_margins[3])))
    # ^ width and height with margins!
    mask2_pil = Image.new('L', (w, h), 0)
    
    px = mask2_pil.load()
    hw_px = trans_len_rel * min(w, h)
    hw_px = int(round(hw_px))
    
    # smooth along the 4 edges
    for i in range(w):
        for j in range(h):
            dist_px = min(i, w - i, j, h - j)
            #print(i, j, px[i,j], dist_px)
            add = 255 * np.exp(-dist_px / hw_px)
            add = int(round(add))
            #px[i,j] = (px[i,j][0] + add, px[i,j][1] + add, px[i,j][2] + add)
            px[i,j] += add
    
    #mask2_pil.show()
    return mask2_pil
    
    
def dict_to_tf_example(image_subdirectory, data, label_map_dict, ignore_difficult_instances=False):
    """
    Convert the dict derived from aml to tf.train.Example proto.
    
    Note that this function normalizes the bounding box coordinates provided
    by the raw data.
    
    Args:
        image_subdirectory: The directory that contains the images
        data: Dictionary returned by aml_dict_to_data_dict()
        label_map_dict: A map from string label names to integers ids
        ignore_difficult_instances: Whether to skip difficult instances in the
            dataset (default = False)
    
    Returns:
        example: The converted tf.train.Example.
        (None if None is passed as the 'data' argument)
    
    Raises:
        ValueError: if the image pointed to by data['filename'] is not a valid JPEG or PNG
    """
    if not data:
        return None
    
    img_path = os.path.join(image_subdirectory, data['filename'])
    if not os.path.isfile(img_path):
        img_path = data['filename']
    
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    #print('image.format', image.format)
    if image.format not in ['JPEG', 'PNG']:
        raise ValueError('Image format is not JPEG and not PNG')
    
    #img_format = 'jpeg'
    #if image.format == 'PNG':
    #    img_format = 'png'
    
    key = hashlib.sha256(encoded_jpg).hexdigest()
    
    width = int(data['size']['width'])
    height = int(data['size']['height'])
    
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    masks = []
    rel_margins = [0.1, 0.1, 0.1, 0.1]
    # ^ [left, upper, right, lower]
    # ^ the values can be set to some random values around the mean 0.1
    trans_len_rel = [0.1,]
    difficult_obj = []
    for obj in data['object']:
        difficult = bool(int(obj['difficult']))
        if ignore_difficult_instances and difficult:
            continue
        
        difficult_obj.append(int(difficult))
        
        # let's not include mask to tfrecord
        '''
        w = obj['bndbox']['xmax'] - obj['bndbox']['xmin']
        h = obj['bndbox']['ymax'] - obj['bndbox']['ymin']
        mask = create_mask_simple_rectangle(w, h, rel_margins, trans_len_rel[0])
        imgByteArr = io.BytesIO()
        mask.save(imgByteArr, format='JPEG')
        imgByteArr = imgByteArr.getvalue()
        masks.append(imgByteArr)
        '''
        
        xmin.append(float(obj['bndbox']['xmin']) / width)
        ymin.append(float(obj['bndbox']['ymin']) / height)
        xmax.append(float(obj['bndbox']['xmax']) / width)
        ymax.append(float(obj['bndbox']['ymax']) / height)
        class_name = obj['name']
        classes_text.append(class_name.encode('utf8'))
        classes.append(label_map_dict[LABELS_IV_TO_MSCOCO[class_name]])
        truncated.append(int(obj['truncated']))
        poses.append(obj['pose'].encode('utf8'))
    
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        #'image/object/mask': dataset_util.bytes_list_feature(masks),
        #'image/bbox_rel_margins': dataset_util.float_list_feature(rel_margins),
        #'image/trans_len_rel': dataset_util.float_list_feature(trans_len_rel),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example


def create_tf_record(data_dir, amls_list, label_map_dict, outpath):
    """
    Write the TFRecord file.
    
    Args:
        data_dir: The directory that contains the images
        amls_list: The of (paths to) *.aml files
        label_map_dict: A map from string label names to integers ids
        outpath: The path to the output file that will be created
    
    Returns:
        None
    """
    
    random.shuffle(amls_list)
    
    writer = tf.python_io.TFRecordWriter(outpath)
    for aml_path in amls_list:
        aml_dict = aml_to_dict(aml_path)
        #print(aml_dict)
        data_dict = aml_dict_to_data_dict(aml_path, aml_dict)
        tf_example = dict_to_tf_example(data_dir, data_dict, label_map_dict)
        #print(type(tf_example))
        if tf_example:
            writer.write(tf_example.SerializeToString())
        #print()
        
    writer.close()


def main(_):
    """
    This function contains all actual work that will be done 
    by tf.app.run() call in '__main__'
    """
    
    #print('\tFLAGS.label_map_path:', FLAGS.label_map_path)
    #print('\tFLAGS.data_dir: "%s"' % FLAGS.data_dir)
    #print('\tFLAGS.output_path: "%s"' % FLAGS.output_path)
    
    if not os.path.isdir(FLAGS.data_dir):
        print("\tthe specified data_dir does not exist")
        return
    
    if not FLAGS.output_path:
        #print("\toutput_path is not specified")
        if FLAGS.data_dir.endswith('/'):
            FLAGS.data_dir = FLAGS.data_dir[:-1]
        FLAGS.output_path = os.path.split(FLAGS.data_dir)[1] + '.tfrecord'
    
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    
    amls_list = []
    for root, dirs, files in os.walk(FLAGS.data_dir):
        for file in files:
            if file.endswith(".aml"):
                aml_path = os.path.join(root, file)
                amls_list.append(aml_path)
    
    create_tf_record(FLAGS.data_dir, amls_list, label_map_dict, FLAGS.output_path)
    
    

#----------------------------------------------------------------
if __name__ == '__main__':
    tf.app.run()
