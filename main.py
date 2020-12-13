from utils import *
from skimage.segmentation import slic
import os
import glob 
from data_prep import *

def get_superpixel(sup_image, segment_val):
    h,w = sup_image.shape
    x, y =[],[]
    if segment_val in sup_image:
        for i in range(h):
            for j in range(w):
                if sup_image[i,j] == segment_val:
                    x.append(j)
                    y.append(i)
        return min(y), min(x), max(y), max(x)
    else:
        return -1, -1, -1, -1

def get_majority_class(seg_gt, bbox, seg_label_list):
    # bbox : [h_start, w_start, h_end, w_end]
    counter = []
    for s in seg_label_list:
        temp = {'id':s['id'], 'name':s['name'], 'rgb':np.array(s['rgb_values']), 'count':0}
        counter.append(temp) 
    h_start, w_start, h_end, w_end = bbox
    seg_patch = seg_gt[h_start:h_end, w_start:w_end]
    h, w, _ = seg_patch.shape
    for i in range(h):
        for j in range(w):
            for c in counter:
                if (c['rgb'] == seg_patch[i,j]).all():
                    c['count'] += 1
    max_ = 0
    max_id = 0
    for index, c in enumerate(counter):
        if c['count'] > max_:
            max_id = c['id']
            max_ = c['count']
    return max_id

def write_row(file, image_name, bbox, class_id):
    file.write(image_name)
    file.write(","+str(bbox[0]))
    file.write(","+str(bbox[1]))
    file.write(","+str(bbox[2]))
    file.write(","+str(bbox[3]))
    file.write(","+str(class_id))
    file.write("\n")
    pass

def get_images(image_name, gt_name):
    image_sample = cv2.imread(image_name)
    image_sample = cv2.cvtColor(image_sample, cv2.COLOR_BGR2RGB)
    seg_sample = cv2.imread(gt_name)
    seg_sample = cv2.cvtColor(seg_sample, cv2.COLOR_BGR2RGB)
    segments_sample = slic(image_sample, n_segments=100, compactness=10)
    return image_sample, seg_sample, segments_sample

image_names = glob.glob("MSRC_ObjCategImageDatabase_v1/*_s.bmp")
gt_names = glob.glob("MSRC_ObjCategImageDatabase_v1/*_s_GT.bmp")

# CSV
# input image name , bbox , gt_val - class
data_file = open("Data.csv",'w')

for image_number in range(len(image_names)):
    image, gt_image, slic_image = get_images(image_names[image_number], gt_names[image_number])
    print("Running image " + str(image_number))
    for pixel in range(256):
        bbox = get_superpixel(slic_image, pixel)
        h1, w1, h2, w2 = bbox
        if h1!=-1:
            # color = (255, 0, 0) 
            # thickness = 2
            class_id = get_majority_class(gt_image, bbox, SEG_LABELS_LIST_v1)
            # image = cv2.rectangle(image_sample, (w1,h1), (w2,h2), color, thickness)
            write_row(data_file, image_names[image_number], bbox, class_id)
            
data_file.close()

