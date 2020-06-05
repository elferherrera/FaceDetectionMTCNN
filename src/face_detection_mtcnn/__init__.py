"""
Face detection function
It loads the networks used for face detections and returns
the bounding boxes for the detected faces
"""

import numpy as np

import torch
from torch.nn.functional import interpolate
from torchvision.ops.boxes import batched_nms

# Importing Neural networks weights
from face_detection_mtcnn.models.mtcnn import PNet, ONet, RNet

# Loading partial MTCNN networks
P_NET = PNet(pretrained=True)
R_NET = RNet(pretrained=True)
O_NET = ONet(pretrained=True)

# Selecting the available device for calculations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seting the networks to eval mode and not training
# This avoids auto grad from being calculated
P_NET = P_NET.eval()
R_NET = R_NET.eval()
O_NET = O_NET.eval()

# Ratio factor used by the detections
FACTOR = 0.709

def detect_face(imgs, minsize=20, threshold=(0.6, 0.7, 0.7)):
    """
    Face detection function
    Given an image or an array of images with the same size
    the bounding boxes for the detected faces are returned

    Important. All images have to have their color channel as RGB

    inputs:
        imgs: image or array of images
        minsize: minimum face size to be found in image
        threshold: thresholds for the neural networks

    output:
        boxes: bounding boxes where faces where found
        probs: probabilities that the found face is a face
        points: points that represent a face
    """

    if isinstance(imgs, (np.ndarray, torch.Tensor)):
        imgs = torch.as_tensor(imgs, device=DEVICE)

        if len(imgs.shape) == 3:
            imgs = imgs.unsqueeze(0)

    else:
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]

        if any(img.size != imgs[0].size for img in imgs):
            raise Exception("MTCNN batch processing only compatible with equal-dimension images.")

        imgs = np.stack([np.uint8(img) for img in imgs])

    # Moving all images to a torch tensor and to the available device
    imgs = torch.as_tensor(imgs, device=DEVICE)

    # Changing dimensions of images if they are as a batch
    # In pytorch the number of images has to be the first
    # dimension
    model_dtype = next(P_NET.parameters()).dtype
    imgs = imgs.permute(0, 3, 1, 2).type(model_dtype)

    batch_size = len(imgs)
    height, width = imgs.shape[2:4]

    min_factor = 12.0 / minsize
    minl = min(height, width)
    minl = minl * min_factor

    # Create scale pyramid
    scale_i = min_factor
    scales = []

    while minl >= 12:
        scales.append(scale_i)
        scale_i = scale_i * FACTOR
        minl = minl * FACTOR

    # First stage
    boxes = []
    image_inds = []
    all_inds = []
    all_i = 0

    for scale in scales:
        im_data = imresample(imgs, (int(height * scale + 1), int(width * scale + 1)))
        im_data = (im_data - 127.5) * 0.0078125

        with torch.no_grad():
            reg, probs = P_NET(im_data)

        boxes_scale, image_inds_scale = generate_bounding_box(reg, probs[:, 1], scale, threshold[0])
        boxes.append(boxes_scale)
        image_inds.append(image_inds_scale)
        all_inds.append(all_i + image_inds_scale)
        all_i += batch_size

    boxes = torch.cat(boxes, dim=0)
    image_inds = torch.cat(image_inds, dim=0).cpu()
    all_inds = torch.cat(all_inds, dim=0)

    # NMS within each scale + image
    pick = batched_nms(boxes[:, :4], boxes[:, 4], all_inds, 0.5)
    boxes, image_inds = boxes[pick], image_inds[pick]

    # NMS within each image
    pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
    boxes, image_inds = boxes[pick], image_inds[pick]

    regw = boxes[:, 2] - boxes[:, 0]
    regh = boxes[:, 3] - boxes[:, 1]
    qq1 = boxes[:, 0] + boxes[:, 5] * regw
    qq2 = boxes[:, 1] + boxes[:, 6] * regh
    qq3 = boxes[:, 2] + boxes[:, 7] * regw
    qq4 = boxes[:, 3] + boxes[:, 8] * regh

    boxes = torch.stack([qq1, qq2, qq3, qq4, boxes[:, 4]]).permute(1, 0)
    boxes = rerec(boxes)

    y, ey, x, ex = pad(boxes, width, height)

    # Second stage
    if len(boxes) > 0:
        im_data = []

        for k, _ in enumerate(y):
            if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                img_k = imgs[image_inds[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
                im_data.append(imresample(img_k, (24, 24)))

        im_data = torch.cat(im_data, dim=0)
        im_data = (im_data - 127.5) * 0.0078125

        with torch.no_grad():
            out = R_NET(im_data)

        out0 = out[0].permute(1, 0)
        out1 = out[1].permute(1, 0)

        score = out1[1, :]
        ipass = score > threshold[1]
        boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)

        image_inds = image_inds[ipass]
        mv_val = out0[:, ipass].permute(1, 0)

        # NMS within each image
        pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)

        boxes, image_inds, mv_val = boxes[pick], image_inds[pick], mv_val[pick]
        boxes = bbreg(boxes, mv_val)
        boxes = rerec(boxes)

    # Third stage
    points = torch.zeros(0, 5, 2, device=DEVICE)
    if len(boxes) > 0:
        y, ey, x, ex = pad(boxes, width, height)

        im_data = []
        for k, _ in enumerate(y):
            if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                img_k = imgs[image_inds[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
                im_data.append(imresample(img_k, (48, 48)))

        im_data = torch.cat(im_data, dim=0)
        im_data = (im_data - 127.5) * 0.0078125

        with torch.no_grad():
            out = O_NET(im_data)

        out0 = out[0].permute(1, 0)
        out1 = out[1].permute(1, 0)
        out2 = out[2].permute(1, 0)

        score = out2[1, :]
        points = out1

        ipass = score > threshold[2]
        points = points[:, ipass]
        boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
        image_inds = image_inds[ipass]
        mv_val = out0[:, ipass].permute(1, 0)

        w_i = boxes[:, 2] - boxes[:, 0] + 1
        h_i = boxes[:, 3] - boxes[:, 1] + 1

        points_x = w_i.repeat(5, 1) * points[:5, :] + boxes[:, 0].repeat(5, 1) - 1
        points_y = h_i.repeat(5, 1) * points[5:10, :] + boxes[:, 1].repeat(5, 1) - 1
        points = torch.stack((points_x, points_y)).permute(2, 1, 0)

        boxes = bbreg(boxes, mv_val)

        # NMS within each image using "Min" strategy
        # pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
        pick = batched_nms_numpy(boxes[:, :4], boxes[:, 4], image_inds, 0.7, "Min")
        boxes, image_inds, points = boxes[pick], image_inds[pick], points[pick]

    boxes = boxes.cpu().numpy()
    points = points.cpu().numpy()

    batch_boxes = []
    batch_points = []
    for b_i in range(batch_size):
        b_i_inds = np.where(image_inds == b_i)
        batch_boxes.append(boxes[b_i_inds].copy())
        batch_points.append(points[b_i_inds].copy())

    batch_boxes, batch_points = np.array(batch_boxes), np.array(batch_points)

    boxes, probs, points = [], [], []
    for box, point in zip(batch_boxes, batch_points):
        box = np.array(box)
        point = np.array(point)

        if len(box) == 0:
            boxes.append(None)
            probs.append([None])
            points.append(None)

        else:
            boxes.append(box[:, :4])
            probs.append(box[:, 4])
            points.append(point)

    boxes = np.array(boxes)
    probs = np.array(probs)
    points = np.array(points)

    # Reducing dimensions of tensor when only one image was used
    if boxes.shape[0] == 1:
        boxes = boxes[0]
        probs = probs[0]
        points = points[0]

    # Changing arrays to lists and integers for easier manipulation
    # when drawing and selecting areas
    boxes = boxes.astype("int").tolist()
    probs = probs.astype("float").tolist()
    points = points.astype("int").tolist()

    return boxes, probs, points


def bbreg(boundingbox, reg):
    """
    Calculating bounding box regions
    """
    if reg.shape[1] == 1:
        reg = torch.reshape(reg, (reg.shape[2], reg.shape[3]))

    width = boundingbox[:, 2] - boundingbox[:, 0] + 1
    height = boundingbox[:, 3] - boundingbox[:, 1] + 1

    b_1 = boundingbox[:, 0] + reg[:, 0] * width
    b_2 = boundingbox[:, 1] + reg[:, 1] * height
    b_3 = boundingbox[:, 2] + reg[:, 2] * width
    b_4 = boundingbox[:, 3] + reg[:, 3] * height
    boundingbox[:, :4] = torch.stack([b_1, b_2, b_3, b_4]).permute(1, 0)

    return boundingbox


def generate_bounding_box(reg, probs, scale, thresh):
    """
    Creates bounding box for region
    """

    stride = 2
    cellsize = 12

    reg = reg.permute(1, 0, 2, 3)

    mask = probs >= thresh
    mask_inds = mask.nonzero()
    image_inds = mask_inds[:, 0]
    score = probs[mask]
    reg = reg[:, mask].permute(1, 0)

    b_b = mask_inds[:, 1:].type(reg.dtype).flip(1)
    q_1 = ((stride * b_b + 1) / scale).floor()
    q_2 = ((stride * b_b + cellsize - 1 + 1) / scale).floor()

    boundingbox = torch.cat([q_1, q_2, score.unsqueeze(1), reg], dim=1)

    return boundingbox, image_inds


def nms_numpy(boxes, scores, threshold, method):
    """
    Non max supressor
    """
    if boxes.size == 0:
        return np.empty((0, 3))

    x_1 = boxes[:, 0].copy()
    y_1 = boxes[:, 1].copy()
    x_2 = boxes[:, 2].copy()
    y_2 = boxes[:, 3].copy()

    area = (x_2 - x_1 + 1) * (y_2 - y_1 + 1)

    index_sorted = np.argsort(scores)
    pick = np.zeros_like(scores, dtype=np.int16)
    counter = 0

    while index_sorted.size > 0:
        i = index_sorted[-1]
        pick[counter] = i
        counter += 1
        idx = index_sorted[0:-1]

        xx1 = np.maximum(x_1[i], x_1[idx]).copy()
        yy1 = np.maximum(y_1[i], y_1[idx]).copy()
        xx2 = np.minimum(x_2[i], x_2[idx]).copy()
        yy2 = np.minimum(y_2[i], y_2[idx]).copy()

        width = np.maximum(0.0, xx2 - xx1 + 1).copy()
        height = np.maximum(0.0, yy2 - yy1 + 1).copy()

        inter = width * height

        if method == "Min":
            o_val = inter / np.minimum(area[i], area[idx])

        else:
            o_val = inter / (area[i] + area[idx] - inter)

        index_sorted = index_sorted[np.where(o_val <= threshold)]

    pick = pick[:counter].copy()

    return pick


def batched_nms_numpy(boxes, scores, idxs, threshold, method):
    """
    Batched nms for numpy arrays
    """

    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=DEVICE)

    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    boxes_for_nms = boxes_for_nms.cpu().numpy()

    scores = scores.cpu().numpy()
    keep = nms_numpy(boxes_for_nms, scores, threshold, method)

    return torch.as_tensor(keep, dtype=torch.long, device=DEVICE)


def pad(boxes, width, height):
    """
    Padding for found boxes
    """

    boxes = boxes.trunc().int().cpu().numpy()
    x_val = boxes[:, 0]
    y_val = boxes[:, 1]
    e_x = boxes[:, 2]
    e_y = boxes[:, 3]

    x_val[x_val < 1] = 1
    y_val[y_val < 1] = 1
    e_x[e_x > width] = width
    e_y[e_y > height] = height

    return y_val, e_y, x_val, e_x


def rerec(bbox_a):
    """
    Box rectangle
    """
    height = bbox_a[:, 3] - bbox_a[:, 1]
    width = bbox_a[:, 2] - bbox_a[:, 0]

    max_val = torch.max(width, height)

    bbox_a[:, 0] = bbox_a[:, 0] + width * 0.5 - max_val * 0.5
    bbox_a[:, 1] = bbox_a[:, 1] + height * 0.5 - max_val * 0.5
    bbox_a[:, 2:4] = bbox_a[:, :2] + max_val.repeat(2, 1).permute(1, 0)

    return bbox_a


def imresample(img, size):
    """
    Image resample
    """
    im_data = interpolate(img, size=size, mode="area")

    return im_data
