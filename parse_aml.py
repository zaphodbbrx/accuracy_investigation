import os
from collections import OrderedDict

import xmltodict

import tensorflow as tf


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
    # print(di)
    # b = xmltodict.unparse(di, pretty=True, indent='  ')
    # print(b)
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

    # print("  -> ", aml_path)
    data = OrderedDict.fromkeys(['filename', 'size', 'object'])
    if 'DataSource' in aml_dict['IvLog']['Parameters']:
        # print("  -> DataSource tag is found")
        data['filename'] = aml_dict['IvLog']['Parameters']['DataSource']
    else:
        # print("  -> DataSource tag is NOT FOUND")
        (dirname, basename) = os.path.split(aml_path)
        (namenoext, ext) = os.path.splitext(basename)
        for ext in ['.png', '.jpg', 'jpeg']:
            attempt = os.path.join(dirname, namenoext + ext)
            if os.path.isfile(attempt):
                # print("  -> '%s' file found" % ext)
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
            li = [li, ]
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