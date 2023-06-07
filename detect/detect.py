import torch.nn.functional as F
from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import numpy as np
import argparse
import sys
sys.path.append('/home/silee/workspace/kroc/classification_dn')
from dn_model import DNClassifier
from grad_cam import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################### Parser ###############################################

parser = argparse.ArgumentParser(description='PyTorch Kaist Pedestrian Transformer Detector')
parser.add_argument('--checkpoint', default=None, type=str, help='checkpoint')
parser.add_argument('--save_path',  default=None, type=str, help='save path')
parser.add_argument('--save_name',  default=None, type=str, help='file name')

args = parser.parse_args()

# Load model checkpoint
checkpoint = torch.load(args.checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Load dn classification model
dn_checkpoint = '/home/silee/workspace/kroc/classification_dn/checkpoint.pth.tar'
dn_model = DNClassifier(n_classes=2)
dn_model.load_state_dict(torch.load(dn_checkpoint))
dn_model.to(device)


# Transforms
transform = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
    )


with open('/home/silee/workspace/kroc/classification_dn/label_map.json', 'r') as j:
    dn_label_map = json.load(j)

def dn_prediction(images, dn_model):
    hypothesis = dn_model(images)
    prob = F.softmax(hypothesis, dim=1) 
    day_prob, night_prob = torch.tensor_split(prob, 2, dim=1)
    day_prob = day_prob.detach().cpu().numpy()
    
    return day_prob

def detect(visible_image, lwir_image, min_score, max_overlap, top_k, image_index, gt_boxes, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """
    # original image
    original_image = visible_image

    # Transform
    visible_image, lwir_image = transform(visible_image), transform(lwir_image)
    
    # Move to default device
    visible_image, lwir_image = visible_image.to(device), lwir_image.to(device)

    # dn predictor
    d_prob = dn_prediction(visible_image.unsqueeze(0), dn_model)

    # Forward prop.
    predicted_locs, predicted_scores, feature_map = model(visible_image.unsqueeze(0), lwir_image.unsqueeze(0), d_prob) # [1, 8732, 4], [1, 8732, 2]

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                                    max_overlap=max_overlap, top_k=top_k)
    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found
    if det_labels == ['person?']:
        return [{'image_id': image_index, 'category_id': rev_label_map[det_labels[0]], 'bbox': det_boxes.squeeze().tolist(), 'score': det_scores[0].item()}]
        
    # Annotate
    # annotated_image = original_image
    # draw = ImageDraw.Draw(annotated_image)
    # font = ImageFont.load_default()

    prediction = list()

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue
                
        # Boxes
        box_location = det_boxes[i].tolist()
        box_location_to_xywh = [box_location[0], box_location[1], box_location[2] - box_location[0],
                                box_location[3] - box_location[1]]

        prediction.append({'image_id': image_index, 'category_id': rev_label_map[det_labels[0]], 'bbox': box_location_to_xywh, 'score': det_scores[0][i].item()})

    return prediction


if __name__ == '__main__':
    json_path = '/home/silee/workspace/kroc/kaistPD_json'

    with open(os.path.join(json_path, 'TEST_images_visible.json'), 'r') as f:
        visible_image_list = json.load(f)

    with open(os.path.join(json_path, 'TEST_images_lwir.json'), 'r') as f:
        lwir_image_list = json.load(f)

    with open(os.path.join(json_path, 'TEST_objects.json'), 'r') as f:
        gt_json_data = json.load(f)

    detected_objects = list()
    for id, (visible_img, lwir_img) in enumerate(tqdm(zip(visible_image_list, lwir_image_list))):
        visible = Image.open(visible_img)
        visible = visible.convert('RGB')

        lwir = Image.open(lwir_img)
        lwir = lwir.convert('RGB')

        detected_object = detect(visible, lwir, min_score=0.2, max_overlap=0.5, top_k=200, image_index=id, gt_boxes=gt_json_data[id]['bbox'])        
        detected_objects.extend(detected_object)

    with open(args.save_path + args.save_name, 'w') as f:
        json.dump(detected_objects, f, indent=4)
