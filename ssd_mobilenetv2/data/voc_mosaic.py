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
                #img_id = self.ids[index]
                imgs, annots = [], []
                img_id = self.ids[index]
                img = cv2.imread(self._imgpath % img_id)
                #height, width, _ = img.shape
            
                imgs.append(img)
                target = ET.parse(self._annopath % img_id).getroot()
              
                annot = self.target_transform(target, 1, 1)
                annot = np.asarray(annot)
                annots.append(annot)
                
                index_list, index = [index], index
                for _ in range(3):
                    while index in index_list:
                        index = np.random.randint(0, len(self.ids))
                
                    index_list.append(index)
                    img = cv2.imread(self._imgpath % self.ids[index])
                    imgs.append(img)
                    target = ET.parse(self._annopath % self.ids[index]).getroot()
                    annot = self.target_transform(target, 1, 1)
                    annot = np.asarray(annot)
                    annots.append(annot)
       
                # 第1，2，3，4张图片按顺时针方向排列，1为左上角图片，先计算出第2张图片的scale，然后推算出其他图片的最大resize尺寸，为了不让四张图片中某几张图片太小造成模型学习困难，scale限制为在0.25到0.75之间生成的随机浮点数。
                scale1 = np.random.uniform(0.2, 0.8)
                height1, width1, _ = imgs[0].shape

                imgs[0] = cv2.resize(imgs[0],
                                    (int(width1 * scale1), int(height1 * scale1)))

                max_height2, max_width2 = int(
                    height1 * scale1), width1 - int(width1 * scale1)
                height2, width2, _ = imgs[1].shape
                scale2 = max_height2 / height2
                if int(scale2 * width2) > max_width2:
                    scale2 = max_width2 / width2
                imgs[1] = cv2.resize(imgs[1],
                                    (int(width2 * scale2), int(height2 * scale2)))

                max_height3, max_width3 = height1 - int(
                    height1 * scale1), width1 - int(width1 * scale1)
                height3, width3, _ = imgs[2].shape
                scale3 = max_height3 / height3
                if int(scale3 * width3) > max_width3:
                    scale3 = max_width3 / width3
                imgs[2] = cv2.resize(imgs[2],
                                    (int(width3 * scale3), int(height3 * scale3)))

                max_height4, max_width4 = height1 - int(height1 * scale1), int(
                    width1 * scale1)
                height4, width4, _ = imgs[3].shape
                scale4 = max_height4 / height4
                if int(scale4 * width4) > max_width4:
                    scale4 = max_width4 / width4
                imgs[3] = cv2.resize(imgs[3],
                                    (int(width4 * scale4), int(height4 * scale4)))

                # 最后图片大小和原图一样
                final_image = np.zeros((height1, width1, 3))
                final_image[0:int(height1 * scale1),
                            0:int(width1 * scale1)] = imgs[0]
                final_image[0:int(height2 * scale2),
                            int(width1 * scale1):(int(width1 * scale1) +
                                                int(width2 * scale2))] = imgs[1]
                final_image[int(height1 * scale1):(int(height1 * scale1) +
                                                int(height3 * scale3)),
                            int(width1 * scale1):(int(width1 * scale1) +
                                                int(width3 * scale3))] = imgs[2]
                final_image[int(height1 * scale1):(int(height1 * scale1) +
                                                int(height4 * scale4)),
                            0:int(width4 * scale4)] = imgs[3]
                
                
                #print(annots.size())
                #print(annots[0][:, :4])
                annots[0][:, :4] *= scale1
                #print("====================")
                #print(annots[0][:, :4])
                annots[1][:, :4] *= scale2
                annots[2][:, :4] *= scale3
                annots[3][:, :4] *= scale4

                annots[1][:, 0] += int(width1 * scale1)
                annots[1][:, 2] += int(width1 * scale1)

                annots[2][:, 0] += int(width1 * scale1)
                annots[2][:, 2] += int(width1 * scale1)
                annots[2][:, 1] += int(height1 * scale1)
                annots[2][:, 3] += int(height1 * scale1)

                annots[3][:, 1] += int(height1 * scale1)
                annots[3][:, 3] += int(height1 * scale1)

                for i in range(4):
                    #pts = ['xmin', 'ymin', 'xmax', 'ymax']
                    for j in range(4):                    
                        # scale height or width
                        annots[i][:, j] = annots[i][:, j] / width1 if j % 2 == 0 else annots[i][:, j] / height1
                        #print(annots[i][:, j])
                #print(height1, width1)
                #print(annots)
                img = final_image 

                height, width = height1, width1
                target = np.concatenate(
                    (annots[0], annots[1], annots[2], annots[3]), axis=0)
                #print(target)

                if self.transform_mosaic is not None:
                    target = np.array(target)
                    img, boxes, labels = self.transform_mosaic(img, target[:, :4], target[:, 4])
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
