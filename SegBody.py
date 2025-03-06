from transformers import pipeline
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw

# Initialize face detection
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Initialize segmentation pipeline
segmenter = pipeline(model="mattmdjaga/segformer_b2_clothes")

def remove_face(img, mask):
    img_arr = np.asarray(img)
    faces = app.get(img_arr)
    faces = faces[0]['bbox']
    w = faces[2] - faces[0]
    h = faces[3] - faces[1]
    faces[0] = faces[0] - (w*0.5)
    faces[2] = faces[2] + (w*0.5)
    faces[1] = faces[1] - (h*0.5)
    faces[3] = faces[3] + (h*0.2)
    face_locations = [(faces[0], faces[1]), (faces[2], faces[3])]
    img1 = ImageDraw.Draw(mask)
    img1.rectangle(face_locations, fill=0)
    return mask

def segment_body(original_img, face=True):
    img = original_img.copy()
    segments = segmenter(img)
    segment_include = ["Hat", "Hair", "Sunglasses", "Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Face", "Left-leg", "Right-leg", "Left-arm", "Right-arm", "Bag", "Scarf"]
    mask_list = [s['mask'] for s in segments if s['label'] in segment_include]
    final_mask = np.array(mask_list[0])
    for mask in mask_list:
        final_mask += np.array(mask)
    final_mask = Image.fromarray(final_mask)
    if not face:
        final_mask = remove_face(img.convert('RGB'), final_mask)
    img.putalpha(final_mask)
    return img, final_mask

def segment_torso(original_img):
    img = original_img.copy()
    segments = segmenter(img)
    segment_include = ["Upper-clothes", "Dress", "Belt", "Face", "Left-arm", "Right-arm"]
    mask_list = [s['mask'] for s in segments if s['label'] in segment_include]
    final_mask = np.array(mask_list[0])
    for mask in mask_list:
        final_mask += np.array(mask)
    final_mask = Image.fromarray(final_mask)
    final_mask = remove_face(img.convert('RGB'), final_mask)
    img.putalpha(final_mask)
    return img, final_mask

def segment_lower_body(original_img):
    img = original_img.copy()
    segments = segmenter(img)
    segment_include = ["Skirt", "Pants", "Left-leg", "Right-leg", "Left-shoe", "Right-shoe"]
    mask_list = [s['mask'] for s in segments if s['label'] in segment_include]
    final_mask = np.array(mask_list[0])
    for mask in mask_list:
        final_mask += np.array(mask)
    final_mask = Image.fromarray(final_mask)
    img.putalpha(final_mask)
    return img, final_mask
