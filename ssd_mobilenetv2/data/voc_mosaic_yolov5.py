"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt

# VOC_CLASSES = (  # always index 0
#     'aeroplane', 'bicycle', 'bird', 'boat',
#     'bottle', 'bus', 'car', 'cat', 'chair',
#     'cow', 'diningtable', 'dog', 'horse',
#     'motorbike', 'person', 'pottedplant',
#     'sheep', 'sofa', 'train', 'tvmonitor')

VOC_CLASSES = (  # always index 0
    'car',)
# VOC_CLASSES = (  # always index 0
#     'car',)
# note: if you used our download scripts, this should be right
VOC_ROOT = osp.join(HOME, "data/")


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []

        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(float(bbox.find(pt).text)) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)

            label_idx = self.class_to_ind[name]
            #print(label_idx)

            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 image_sets=[('2007', 'train')],#, ('2012', 'trainval')],
                 transform=None,
                 transform_mosaic=None, 
                 target_transform=VOCAnnotationTransform(),
                 use_mosaic=False,
                 dataset_name='VOC0712'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.transform_mosaic = transform_mosaic
        self.target_transform = target_transform
        self.use_mosaic = use_mosaic
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        '''
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape
        '''
        if self.use_mosaic:
            #print("=====use_mosaic======")
            if np.random.uniform(0, 1.) < 0.5:
                #print("=====use_mosaic======")
                imgs, annots = [], []
                img_id = self.ids[index]
                img = cv2.imread(self._imgpath % img_id)
                height0, width0, _ = img.shape
                
                imgs.append(img)
                target = ET.parse(self._annopath % img_id).getroot()
              
                annot = self.target_transform(target, width0, height0)
                #print(annot)
                annot = np.asarray(annot)
                annots.append(annot)
                
                index_list, index = [index], index
                for _ in range(3):
                    while index in index_list:
                        index = np.random.randint(0, len(self.ids))
                
                    index_list.append(index)
                    img = cv2.imread(self._imgpath % self.ids[index])
                    height, width, _ = img.shape
                    imgs.append(img)
                    target = ET.parse(self._annopath % self.ids[index]).getroot()
                    annot = self.target_transform(target, width, height)
                    annot = np.asarray(annot)
                    annots.append(annot)

                #print(annots)
                # 第1，2，3，4张图片按顺时针方向排列，1为左上角图片，先计算出第2张图片的scale，然后推算出其他图片的最大resize尺寸，为了不让四张图片中某几张图片太小造成模型学习困难，scale限制为在0.25到0.75之间生成的随机浮点数。
                min_offset = 0.2
        
                h, w , _ = imgs[0].shape
            
                imgs1 = []
                for x in imgs:
                    x = cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC)
                    imgs1.append(x)
        
                cut_x = np.random.randint(int(w*min_offset), int(w*(1 - min_offset)))
                cut_y = np.random.randint(int(h*min_offset), int(h*(1 - min_offset)))

               
                d1 = imgs1[0][:cut_y, :cut_x, :]  ##h*w
                d2 = imgs1[1][cut_y:, :cut_x, :]
                d3 = imgs1[2][cut_y:, cut_x:, :]
                d4 = imgs1[3][:cut_y, cut_x:, :]

                tmp1 = np.concatenate((d1, d2), axis=0) ##纵向
                tmp2 = np.concatenate((d4, d3), axis=0)         
                img = np.concatenate((tmp1, tmp2), axis=1)
                #print(img.shape)

                # plt.imshow(img)
                # plt.show()
  
                target = merge_bboxes(annots, cut_x, cut_y, w, h)
                #print(target)

                if self.transform_mosaic and len(target)>=1:
                    target = np.array(target)

                    img, boxes, labels = self.transform_mosaic(img, target[:, :4], target[:, 4])
                    # to rgb
                    height, width, _ = imgs[0].shape
                    img = img[:, :, (2, 1, 0)]
                    # img = img.transpose(2, 0, 1)
                    target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
                
                else:
                    img_id = self.ids[index]
                    target = ET.parse(self._annopath % img_id).getroot()
                    img = cv2.imread(self._imgpath % img_id)
                    height, width, _ = img.shape
                    if self.target_transform is not None:
                        target = self.target_transform(target, width, height)
                    #print(target)
                    if self.transform is not None:
                        target = np.array(target)
                        img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
                        # to rgb
                        img = img[:, :, (2, 1, 0)]
                        # img = img.transpose(2, 0, 1)
                        target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

            else:
                #print("====no mosaic===")
                img_id = self.ids[index]
                target = ET.parse(self._annopath % img_id).getroot()
                img = cv2.imread(self._imgpath % img_id)
                height, width, _ = img.shape

                if self.target_transform is not None:
                    target = self.target_transform(target, width, height)
                    #print(target)
                if self.transform is not None:
                    target = np.array(target)
                    img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
                    # to rgb
                    img = img[:, :, (2, 1, 0)]
                    # img = img.transpose(2, 0, 1)
                    target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        else:

            img_id = self.ids[index]

            target = ET.parse(self._annopath % img_id).getroot()
            img = cv2.imread(self._imgpath % img_id)
            height, width, _ = img.shape

            if self.target_transform is not None:
                target = self.target_transform(target, width, height)
            #print(target)
            if self.transform is not None:
                target = np.array(target)
                img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
                # to rgb
                img = img[:, :, (2, 1, 0)]
                # img = img.transpose(2, 0, 1)
                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))


        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        #print(self._imgpath % img_id)
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


def merge_bboxes(bboxes, cutx, cuty, w, h):
    
    cutx /= w
    cuty /= h
    merge_bbox = []
    for i in range(len(bboxes)):
        for box in bboxes[i]:
            tmp_box = []
            x1, y1, x2, y2, label = box[0], box[1], box[2], box[3], box[4]
            box[0], box[1], box[2], box[3], box[4] = x1, y1, x2-x1, y2-y1, label
            x, y, w, h = x1, y1, x2-x1, y2-y1
            
            if i == 0:
                if box[1]-box[3]/2 > cuty or box[0]-box[2]/2 > cutx:
                    continue

                if box[1]+box[3]/2 > cuty and box[1]-box[3]/2 < cuty:
                    h -= (box[1]+box[3]/2-cuty)
                    y -= (box[1]+box[3]/2-cuty)/2

                if box[0]+box[2]/2 > cutx and box[0]-box[2]/2 < cutx:
                    w -= (box[0]+box[2]/2-cutx)
                    x -= (box[0]+box[2]/2-cutx)/2
                
            if i == 1:
                if box[1]+box[3]/2 < cuty or box[0]-box[2]/2 > cutx:
                    continue

                if box[1]+box[3]/2 > cutx and box[1]-box[3]/2 < cutx:
                    h -= (cuty-(box[1]-box[3]/2))
                    y += (cuty-(box[1]-box[3]/2))/2
                
                if box[0]+box[2]/2 > cutx and box[0]-box[2]/2 < cutx:
                    w -= (box[0]+box[2]/2-cutx)
                    x -= (box[0]+box[2]/2-cutx)/2

            if i == 2:
                if box[1]+box[3]/2 < cuty or box[0]+box[2]/2 < cutx:
                    continue

                if box[1]+box[3]/2 > cuty and box[1]-box[3]/2 < cuty:
              
                    h -= (cuty-(box[1]-box[3]/2))
                    y += (cuty-(box[1]-box[3]/2))/2

                if box[0]+box[2]/2 > cutx and box[0]-box[2]/2 < cutx:
                    w -= (cutx-(box[0]-box[2]/2))
                    x += (cutx-(box[0]-box[2]/2))/2

            if i == 3:
                if box[1]-box[3]/2 > cuty or box[0]+box[2]/2 < cutx:
                    continue

                if box[1]+box[3]/2 > cuty and box[1]-box[3]/2 < cuty:
                    h -= (box[1]+box[3]/2-cuty)
                    y -= (box[1]+box[3]/2-cuty)/2

                if box[0]+box[2]/2 > cutx and box[0]-box[2]/2 < cutx:
                    w -= (cutx-(box[0]-box[2]/2))
                    x += (cutx-(box[0]-box[2]/2))/2
            
            x11, y11, x22, y22 = x, y, x+w, y + h
            
            tmp_box.append(x11)
            tmp_box.append(y11)
            tmp_box.append(x22)
            tmp_box.append(y22)
            tmp_box.append(box[4])
            merge_bbox.append(tmp_box)

    return merge_bbox