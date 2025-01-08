import numpy as np
from detection import *
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.filters import gaussian

# def read_face_labels(image_paths):
#     label_path = "list_bbox_celeba.txt"
#     n_images = len(image_paths)
#     f = open(label_path, "r")
#     f.readline()
#     f.readline()
#     faces = np.array([],dtype=np.int).reshape(0,4)
#     for line in f:
#         if faces.shape[0]>40:
#             break
#         parts = line.strip().split(' ')
#         parts = list(filter(None, parts))
#         #print(line,parts)
#         image_file = parts[0]   
#         if image_file in image_paths:
#             x_1 = int(parts[1])
#             y_1 = int(parts[2])
#             width = int(parts[3])
#             height = int(parts[4])
#             faces = np.vstack((faces, np.asarray([y_1, x_1, height, width])))
#     return faces
        

def read_facial_labels(image_paths):
    label_path = "list_landmarks_align_celeba.txt"
    n_images = len(image_paths)
    f = open(label_path, "r")
    f.readline()
    f.readline()
    lefteyes = np.array([],dtype=np.int).reshape(0,2)
    righteyes = np.array([],dtype=np.int).reshape(0,2)
    noses = np.array([],dtype=np.int).reshape(0,2)
    mouths = np.array([],dtype=np.int).reshape(0,2)
    for line in f:
        if lefteyes.shape[0]>40:
            break
        parts = line.strip().split(' ')
        parts = list(filter(None, parts))
        #print(line,parts)
        image_file = parts[0]
        if image_file in image_paths:
            lefteye_c = int(parts[1])
            lefteye_r = int(parts[2])
            righteye_c = int(parts[3])
            righteye_r = int(parts[4])
            nose_c = int(parts[5])
            nose_r = int(parts[6])
            leftmouth_c = int(parts[7])
            leftmouth_r = int(parts[8])
            rightmouth_c = int(parts[9])
            rightmouth_r = int(parts[10])
            mouth_c = int((leftmouth_c+rightmouth_c)/2)
            mouth_r = int((leftmouth_r+rightmouth_r)/2)
            
            lefteyes = np.vstack((lefteyes, np.asarray([lefteye_r, lefteye_c])))
            righteyes = np.vstack((righteyes, np.asarray([righteye_r, righteye_c])))
            noses = np.vstack((noses, np.asarray([nose_r, nose_c])))
            mouths = np.vstack((mouths, np.asarray([mouth_r, mouth_c])))
    parts = (lefteyes, righteyes, noses, mouths)
    return parts


def get_detector(part_h, part_w, parts, image_paths):
    n = len(image_paths)
    part_shape = (part_h,part_w)
    avg_part = np.zeros((part_shape))
    for i,image_path in enumerate(image_paths):
        image = io.imread('./face/'+image_path, as_grey=True)
        part_r = parts[i][0]
        part_c = parts[i][1]
        #print(image_path, part_r, part_w, part_r-part_h/2, part_r+part_h/2)
        part_image = image[int(part_r-part_h/2):int(part_r+part_h/2), \
                              int(part_c-part_w/2):int(part_c+part_w/2)]   
        avg_part = np.asarray(part_image)+np.asarray(avg_part)
    avg_part = avg_part/n
    return avg_part


def get_heatmap(image, face_feature, face_shape, detectors_list, parts):
    _, _, _, _, face_response_map = pyramid_score \
        (image, face_feature, face_shape, stepSize = 30, scale = 0.8)
    face_response_map=resize(face_response_map,image.shape)
    face_heatmap_shifted = shift_heatmap(face_response_map, [0,0])
    for i,detector in enumerate(detectors_list):
        part = parts[i]
        max_score, r, c, scale,response_map = pyramid_score\
            (image, face_feature, face_shape,stepSize = 30, scale=0.8)
        mu, std = compute_displacement(part, face_shape)
        response_map = resize(response_map, face_response_map.shape)
        response_map_shifted = shift_heatmap(response_map, mu)
        heatmap = gaussian(response_map_shifted, std)
        face_heatmap_shifted+= heatmap
    return face_heatmap_shifted


def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


