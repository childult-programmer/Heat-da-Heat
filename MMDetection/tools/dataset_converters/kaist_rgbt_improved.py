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


if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
    

INFO = {
    "dataset": "KAIST Multispectral Pedestrian Benchmark",
    "url": "https://soonminhwang.github.io/rgbt-ped-detection/",
    "related_project_url": "http://multispectral.kaist.ac.kr",
    "publish": "CVPR 2015",
    "version": "Sanitzied annotation (2016 BMVC)",
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
_annopath = os.path.join(ori_path, 'annotations-xml-181027', '%s', '%s', '%s.xml')

ori_imgpath = os.path.join(ori_path, 'images', '%s', '%s', '%s', '%s.jpg') 
imgpath = os.path.join('%s', '%s', '%s', '%s.jpg')
    
def get_boxes(target) :
    
    pts = ['x', 'y', 'w', 'h']
    
    res = [ [-1, 0, 0, 0, 0] ]

    for obj in target.iter('object'):           
        name = obj.find('name').text.lower().strip()            
        bbox = obj.find('bndbox')

        #label_idx = OBJ_CLS_TO_IDX[name] if name not in OBJ_IGNORE_CLASSES else -1
        bndbox = [1]
        bndbox = bndbox + [ int(bbox.find(pt).text) for pt in pts ]
        
        res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind, occ]

    return np.array(res, dtype=np.float)

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
            
            target = ET.parse(_annopath % ( set_id, vid_id, img_id ) ).getroot()
            
            boxes = get_boxes(target)
            boxes = boxes[1:]

            if boxes != [] :
                vis = Image.open( ori_imgpath % ( set_id, vid_id, 'visible', img_id ), mode='r' ).convert('RGB')
                vis_info = pycococreatortools.create_image_info(image_id, imgpath % ( set_id, vid_id, 'visible', img_id ), vis.size)
                coco_output["images"].append(vis_info)

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
                
            image_id = image_id + 1
        
    
        print('positive_box_num: ', positive_box_num)
        print('ignore_box_num: ', ignore_box_num)

        with open(data_path + '/' + phase + '.json', 'w') as output_json_file:
            json.dump(coco_output, output_json_file)
    
    else :
        for ii, annotation_path in enumerate(tqdm(ids, desc='Making')):

            set_id = ids[ii][0]
            vid_id = ids[ii][1]
            img_id = ids[ii][2]
            
            target = ET.parse(_annopath % ( set_id, vid_id, img_id ) ).getroot()
            
            boxes = get_boxes(target)
            boxes = boxes[1:]
            
            vis = Image.open( ori_imgpath % ( set_id, vid_id, 'visible', img_id ), mode='r' ).convert('RGB')
            vis_info = pycococreatortools.create_image_info(image_id, imgpath % ( set_id, vid_id, 'visible', img_id ), vis.size)
            coco_output["images"].append(vis_info)
            boxes_gt = boxes
            
            for annotation in boxes_gt:
                annotation = annotation.tolist()
                class_id = 1
                category_info = {'id': class_id, 'is_crowd': False}
                annotation_info = pycococreatortools.create_annotation_info(annotation_id, image_id, category_info, annotation, vis.size)
                if annotation_info is not None:
                    coco_output["annotations"].append(annotation_info)
                annotation_id += 1
                
            image_id = image_id + 1

        with open(data_path + '/' + phase + '.json', 'w') as output_json_file:
            json.dump(coco_output, output_json_file)
                
    
if __name__ == '__main__':
    data_path = 'datasets/kaist-rgbt'
    convert(phase='train', data_path=data_path)
    convert(phase='val', data_path=data_path)
