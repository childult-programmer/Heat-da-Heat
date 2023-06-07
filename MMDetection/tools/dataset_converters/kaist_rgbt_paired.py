import os
import datetime
import json
import numpy as np
import glob
from scipy.io import loadmat
from PIL import Image
from tqdm import tqdm

import sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import pycococreatortools

INFO = {
    "dataset": "KAIST Multispectral Pedestrian Benchmark",
    "url": "https://soonminhwang.github.io/rgbt-ped-detection/",
    "related_project_url": "http://multispectral.kaist.ac.kr",
    "publish": "CVPR 2015",
    "version": "Paired annotation (2019 ICCV)",
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        "id": 1,
        "name": "person",
        'supercategory': 'person',
    },
]

ori_path = '/raid/datasets/kaist-rgbt/'
vis_annopath_KAIST = os.path.join(ori_path, 'annotations_paired', '%s', '%s', 'visible', '%s.txt')
lwir_annopath_KAIST = os.path.join(ori_path, 'annotations_paired', '%s', '%s', 'lwir', '%s.txt')
ori_imgpath = os.path.join(ori_path, 'images', '%s', '%s', '%s', '%s.jpg') 
imgpath = os.path.join('%s', '%s', '%s', '%s.jpg')


def get_boxes(bbox) :
    
    boxes = [[-1, 0, 0, 0, 0]]
    
    if bbox != [] :
        for i in range(len(bbox)) :

            name = bbox[i][0]
            bndbox = [1]        
            bndbox = bndbox + [int(i) for i in bbox[i][1:5]]
            boxes += [bndbox]

    boxes = np.array(boxes, dtype=np.float)
    
    return boxes

def convert(phase, data_path):
    assert phase in ['train', 'val']
    
    if phase == 'train' :
        ids = list() 
        for line in open(os.path.join(data_path, 'train.txt')):
            ids.append((line.strip().split('/')))
    else :
        ids = list() 
        for line in open(os.path.join(data_path, 'val.txt')):
            ids.append((line.strip().split('/')))
            
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    
    coco_output_lwir = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    annotation_id = 1
    annotation_id_lwir = 1

    positive_box_num = 0
    positive_box_num_lwir = 0
    ignore_box_num = 0
    ignore_box_num_lwir = 0
    
    if phase == 'train' :
        
        for ii, annotation_path in enumerate(tqdm(ids, desc='Making')):

            set_id = ids[ii][0]
            vid_id = ids[ii][1]
            img_id = ids[ii][2]

            vis_boxes = list()
            lwir_boxes = list()

            for line in open(vis_annopath_KAIST % ( set_id, vid_id, img_id )) :
                vis_boxes.append(line.strip().split(' '))
            for line in open(lwir_annopath_KAIST % ( set_id, vid_id, img_id)) :
                lwir_boxes.append(line.strip().split(' '))

            vis_boxes = vis_boxes[1:]
            lwir_boxes = lwir_boxes[1:]

            if vis_boxes != [] :
                vis = Image.open( ori_imgpath % ( set_id, vid_id, 'visible', img_id ), mode='r' ).convert('RGB')
                vis_info = pycococreatortools.create_image_info(image_id, imgpath % ( set_id, vid_id, 'visible', img_id ), vis.size)
                coco_output["images"].append(vis_info)
                boxes = get_boxes(vis_boxes)[1:]

                slt_msk = np.logical_and(boxes[:, 0] == 1, boxes[:, 4] >= 50)
                boxes_gt = boxes[slt_msk, 1:5]
                positive_box_num += boxes_gt.shape[0]
                for annotation in boxes_gt:
                    annotation = annotation.tolist()
                    class_id = 1
                    category_info = {'id': class_id, 'is_crowd': False}
                    annotation_info = pycococreatortools.create_annotation_info(annotation_id, image_id, category_info, annotation, vis.size)
                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)
                    annotation_id += 1

                slt_msk = np.logical_or(boxes[:, 0] != 1, boxes[:, 4] < 50)
                boxes_ig = boxes[slt_msk, 1:5]
                ignore_box_num += boxes_ig.shape[0]
                for annotation in boxes_ig:
                    annotation = annotation.tolist()
                    category_info = {'id': 1, 'is_crowd': True}
                    annotation_info = pycococreatortools.create_annotation_info(annotation_id, image_id, category_info, annotation, vis.size)
                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)
                    annotation_id += 1 


            if lwir_boxes != [] :
                lwir = Image.open( ori_imgpath % ( set_id, vid_id, 'lwir', img_id ), mode='r' ).convert('L')
                lwir_info = pycococreatortools.create_image_info(image_id, imgpath % ( set_id, vid_id, 'lwir', img_id ), lwir.size)
                coco_output_lwir["images"].append(lwir_info)
                boxes = get_boxes(lwir_boxes)[1:]

                slt_msk = np.logical_and(boxes[:, 0] == 1, boxes[:, 4] >= 50)
                boxes_gt = boxes[slt_msk, 1:5]
                positive_box_num_lwir += boxes_gt.shape[0]
                for annotation in boxes_gt:
                    annotation = annotation.tolist()
                    class_id = 1
                    category_info = {'id': class_id, 'is_crowd': False}
                    annotation_info = pycococreatortools.create_annotation_info(annotation_id_lwir, image_id, category_info, annotation, lwir.size)
                    if annotation_info is not None:
                        coco_output_lwir["annotations"].append(annotation_info)
                    annotation_id_lwir += 1

                slt_msk = np.logical_or(boxes[:, 0] != 1, boxes[:, 4] < 50)
                boxes_ig = boxes[slt_msk, 1:5]
                ignore_box_num_lwir += boxes_ig.shape[0]
                for annotation in boxes_ig:
                    annotation = annotation.tolist()
                    category_info = {'id': 1, 'is_crowd': True}
                    annotation_info = pycococreatortools.create_annotation_info(annotation_id_lwir, image_id, category_info, annotation, lwir.size)
                    if annotation_info is not None:
                        coco_output_lwir["annotations"].append(annotation_info)
                    annotation_id_lwir += 1 
                
            image_id = image_id + 1
        
    
        print('positive_box_num: ', positive_box_num)
        print('ignore_box_num: ', ignore_box_num)
        print('positive_box_num_lwir: ', positive_box_num_lwir)
        print('ignore_box_num_lwir: ', ignore_box_num_lwir)

        with open(data_path + '/' + phase + '.json', 'w') as output_json_file:
            json.dump(coco_output, output_json_file)
        with open(data_path + '/' + phase + '_lwir.json', 'w') as output_json_file:
            json.dump(coco_output_lwir, output_json_file)
    
    else :
        for ii, annotation_path in enumerate(tqdm(ids, desc='Making')):

            set_id = ids[ii][0]
            vid_id = ids[ii][1]
            img_id = ids[ii][2]
            
            vis_boxes = list()
            lwir_boxes = list()
            
            for line in open(vis_annopath_KAIST % ( set_id, vid_id, img_id )) :
                vis_boxes.append(line.strip().split(' '))
            for line in open(lwir_annopath_KAIST % ( set_id, vid_id, img_id )) :
                lwir_boxes.append(line.strip().split(' '))
                
            vis_boxes = vis_boxes[1:]
            lwir_boxes = lwir_boxes[1:]
            
            vis = Image.open( ori_imgpath % ( set_id, vid_id, 'visible', img_id ), mode='r' ).convert('RGB')
            vis_info = pycococreatortools.create_image_info(image_id, imgpath % ( set_id, vid_id, 'visible', img_id ), vis.size)
            coco_output["images"].append(vis_info)
            boxes_gt = get_boxes(vis_boxes)
            
            for annotation in boxes_gt:
                annotation = annotation.tolist()
                class_id = 1
                category_info = {'id': class_id, 'is_crowd': False}
                annotation_info = pycococreatortools.create_annotation_info(annotation_id, image_id, category_info, annotation, vis.size)
                if annotation_info is not None:
                    coco_output["annotations"].append(annotation_info)
                annotation_id += 1
                
            lwir = Image.open( ori_imgpath % ( set_id, vid_id, 'lwir', img_id ), mode='r' ).convert('L')
            lwir_info = pycococreatortools.create_image_info(image_id, imgpath % ( set_id, vid_id, 'lwir', img_id ), lwir.size)
            coco_output_lwir["images"].append(lwir_info)
            boxes_gt = get_boxes(lwir_boxes)
            
            for annotation in boxes_gt:
                annotation = annotation.tolist()
                class_id = 1
                category_info = {'id': class_id, 'is_crowd': False}
                annotation_info = pycococreatortools.create_annotation_info(annotation_id_lwir, image_id, category_info, annotation, lwir.size)
                if annotation_info is not None:
                    coco_output_lwir["annotations"].append(annotation_info)
                annotation_id_lwir += 1
            
            image_id = image_id + 1

        with open(data_path + '/' + phase + '.json', 'w') as output_json_file:
            json.dump(coco_output, output_json_file)
        with open(data_path + '/' + phase + '_lwir.json', 'w') as output_json_file:
            json.dump(coco_output_lwir, output_json_file)
                
if __name__ == '__main__':
    data_path = 'datasets/kaist-rgbt'
    convert(phase='train', data_path=data_path)
    convert(phase='val', data_path=data_path)
