import json
import os
from PIL import Image
import time
from argparse import ArgumentParser
import shapely.geometry
import numpy as np


def add_info(coco, args):
    coco['info'] = {}
    info_field_set = ['contributor', 'about', 'description',
                      'url', 'version', 'date_created', 'year']
    if args.info_path:
        jf_info = open(args.info_path)
        j_info = json.load(jf_info)
        jf_info.close()
    else:
        j_info = args.__dict__
    for field in info_field_set:
        if field in j_info:
            value = j_info[field]
        else:
            value = args.__getattribute__(field)
        coco['info'][field] = value
    if not coco['info']['date_created']:
        time_str = time.strftime('%d/%m/%Y', time.localtime(int(time.time())))
        coco['info']['date_created'] = time_str
        if not coco['info']['year']:
            coco['info']['year'] = int(time_str.split('/')[-1])
    return coco


def add_others(coco, args):
    coco['images'] = []
    coco['annotations'] = []
    sse_ls = os.listdir(args.sse_folder)
    sse_ls = [os.path.join(args.sse_folder, i) for i in sse_ls]

    img_id = 0
    ann_id = 0
    for sse_fname in sse_ls:
        jf_sse = open(sse_fname)
        json_sse = json.load(jf_sse)
        jf_sse.close()

        imgname = json_sse['file']
        img_path = os.path.join(args.images_folder, imgname)
        img_width, img_height = Image.open(img_path).size
        img_dict = {'id': img_id,
                    'file_name': imgname,
                    'width': img_width,
                    'height': img_height}
        coco['images'].append(img_dict)

        polygons_objs = json_sse['objects']
        for polygon_obj in polygons_objs:
            polygon_coor = polygon_obj['polygon']
            polygon_list = []
            for coor in polygon_coor:
                polygon_list.extend([coor['x'], coor['y']])
            polygon_array = np.array(polygon_list).reshape(-1, 2)
            polygon_shapely = shapely.geometry.Polygon(polygon_array)
            polygon_area = polygon_shapely.area

            # CrowdAI的bbox实在过于窒息不能理解如何计算，决定转换后的bbox遵循xywh，https://blog.csdn.net/zhuoyuezai/article/details/84315113
            x1, y1 = np.min(polygon_array, axis=0)
            x2, y2 = np.max(polygon_array, axis=0)
            w, h = x2 - x1, y2 - y1
            polygon_bbox = [x1, y1, w, h]

            ann_dict = {'id': ann_id,
                        'image_id': img_id,
                        'segmentation': [polygon_list],
                        'area': polygon_area,
                        'bbox': polygon_bbox,
                        'category_id': polygon_obj['classIndex'],
                        'iscrowd': 0}
            coco['annotations'].append(ann_dict)
            ann_id += 1

        img_id += 1

    return coco


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument("--sse_folder", type=str, default='sse',
                      help='input folder, the folder of annotations producted by Semantic Segmentation Editor')
    args.add_argument("--coco_folder", type=str, default='.',
                      help='output folder, the folder of annotations in MS COCO format')
    args.add_argument("--images_folder", type=str, default='images',
                      help='folder of images, images\' information is needed for MS COCO format')
    args.add_argument('--contributor', type=str, default='Halle Astra',
                      help='attribute `contributor` in COCO\'s info field, invalid when `info_path` is given')
    args.add_argument('--about', type=str, default='Dataset for building segmentation',
                      help='attribute `about` in COCO\'s info field, invalid when `info_path` is given')
    args.add_argument('--description', type=str, default='building segmentation dataset',
                      help='attribute `description` in COCO\'s info field, invalid when `info_path` is given')
    args.add_argument('--url', type=str, default='https://github.com/Halle-Astra',
                      help='attribute `url` in COCO\'s info field, invalid when `info_path` is given')
    args.add_argument('--version', type=str, default='1.0',
                      help='attribute `version` in COCO\'s info field, invalid when `info_path` is given')
    args.add_argument('--date_created', type=str, default='',
                      help='attribute `date_created` in COCO\'s info field, '
                           'invalid when `info_path` is given, default is now')
    args.add_argument('--year', type=int, default=0,
                      help='attribute `year` in COCO\'s info field, '
                           'invalid when `info_path` is given or `date_created` is given, default is now')
    args.add_argument('--info_path', type=str, default='',
                      help='values in COCO\'s info field in json format')
    args = args.parse_args()

    coco = {}
    coco = add_info(coco, args)
    coco = add_others(coco, args)

    coco_jf = open('annotation.json', 'w')
    json.dump(coco, coco_jf)
