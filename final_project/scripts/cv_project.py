import os
import numpy as np
from PIL import Image, ImageDraw
import ast
import torch
import sys
import random
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
from matplotlib import pyplot as plt


######################################### Load data#########################################

def Buses_data_opening(path):
    """Input: the path containing 1) busesTrain folder of training images
                                  2)Train_annotations - ground truth annotations file folder
       Output: 1) Buses_as_numpy - a numpy array of size: num_samples*W*H*num_color_channels
               2)annotations - a list of lists where the ith compoenent is a list containing all annotations of the ith image
                    [xmin1, ymin1, width1, height1,color1],..,[xminN, yminN, widthN, heightN,colorN]"""
    images_path = os.path.join(path, "busesTrain")
    images = [os.path.join(images_path, f) for f in os.listdir(images_path)]
    Buses_as_numpy = np.array([np.array(Image.open(fname)) for fname in images])
    filenames = [str(f) for f in os.listdir(images_path)]
    annotations_folder = os.path.join(path, "Train_annotations")
    annotations_filename = os.path.join(annotations_folder,os.listdir(annotations_folder)[0])
    annotations = []
    with open(annotations_filename) as f:
        for line in f:
            image_name, image_rects_string = line.split(':')
            image_rects_list = ast.literal_eval(image_rects_string)
            if type(image_rects_list[0]) == int:
                image_rects_list = [image_rects_list]
            annotations.append(image_rects_list)
    return Buses_as_numpy, annotations , filenames


######################################### Dataset augmentations #########################################

class BusDataset(data.Dataset):
    image_size = 448
    def __init__(self,root,train,transform):
        print('data init')
        self.root= root
        self.train = train
        self.transform= transform
        self.image_size = 448
        self.boxes = []
        self.labels = []
        self.mean = (121,132,145) #RGB - to calc for our dataset
        self.data,self.label_lists,self.fnames = Buses_data_opening(root)

        for i, v in enumerate(self.label_lists):
            # i is image number in datatset, v is a list of all rects in image i
            self.fnames.append(i) #consider adding file name here
            num_boxes = len(v)
            box=[]
            label=[]
            for j in range(num_boxes):
                x = float(v[j][0]) #x upper left
                y = float(v[j][1]) #y upper left
                x2 = float(v[j][0]+v[j][2]) #x lower right
                y2 = float(v[j][1]+v[j][3])  #y lower right
                c = v[j][4] #classes 1-6
                box.append([x,y,x2,y2])
                label.append(int(c))  #classes 1-6
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
        self.num_samples = len(self.boxes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(self.root + '/busesTrain/' + fname))
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()

        if self.train:
            img, boxes = self.random_flip_x(img,boxes)
            img, boxes = self.random_flip_y(img, boxes)
            img, boxes = self.randomScale(img,boxes)
            img = self.add_gaussian_noise(img)
            img = self.add_uniform_noise(img)
            img = self.randomBlur(img)
            img = self.random_bright(img)
        # #debug
        box_show = boxes.numpy().reshape(-1)
        print(box_show)
        img_show = img
        pt1=(int(box_show[0]),int(box_show[1])); pt2=(int(box_show[2]),int(box_show[3]))
        cv2.rectangle(img_show,pt1=pt1,pt2=pt2,color=(0,255,0),thickness=2)
        plt.figure()
        #cv2.rectangle(img,pt1=(10,10),pt2=(100,100),color=(0,255,0),thickness=1)
        plt.imshow(img_show)
        plt.show()
        # #debug
        h,w,_ = img.shape
        boxes /= torch.Tensor([w,h,w,h]).expand_as(boxes)
        img = self.subMean(img, self.mean)
        img = cv2.resize(img, (self.image_size, self.image_size))
        target = self.encoder(boxes, labels)  # 7x7x16
        for t in self.transform:
            img = t(img)
        return img,target

    def encoder(self,boxes,labels):
        '''
        boxes (tensor) [[x1,y1,x2,y2],[]]
        labels (tensor) [...]
        return 7x7x16 B=2
        '''
        grid_num = 14
        target = torch.zeros((grid_num,grid_num,16))
        cell_size = 1./grid_num
        wh = boxes[:,2:]-boxes[:,:2]
        cxcy = (boxes[:,2:]+boxes[:,:2])/2
        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i]
            ij = (cxcy_sample/cell_size).ceil()-1 #for each box finds in which i,j grid cell it is located
            target[int(ij[1]),int(ij[0]),4] = 1
            target[int(ij[1]),int(ij[0]),9] = 1
            target[int(ij[1]),int(ij[0]),int(labels[i])+9] = 1 # puts 1 in the correct class from 6 optional classes
            xy = ij*cell_size # location in each grid cell of the top left corner of the box
            delta_xy = (cxcy_sample -xy)/cell_size
            target[int(ij[1]),int(ij[0]),2:4] = wh[i]
            target[int(ij[1]),int(ij[0]),:2] = delta_xy
            target[int(ij[1]),int(ij[0]),7:9] = wh[i]
            target[int(ij[1]),int(ij[0]),5:7] = delta_xy
        return target

    def random_flip_x(self, im, boxes):
        if random.random() < 0.5:
            print('random_flip_x')
            im_lr = np.fliplr(im).copy()
            h,w,_ = im.shape
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
            return im_lr, boxes
        return im, boxes

    def random_flip_y(self, im, boxes):
        if random.random() < 0.5:
            print('random_flip_y')
            im_lr = np.flipud(im).copy()
            h,w,_ = im.shape
            ymin = h - boxes[:,3]
            ymax = h - boxes[:,1]
            boxes[:,1] = ymin
            boxes[:,3] = ymax
            return im_lr, boxes
        return im, boxes

    def add_gaussian_noise(self,im):
        if random.random() < 0.5:
            print('add_gaussian_noise')
            gaussian_noise = im.copy()
            cv2.randn(gaussian_noise, 0, 30)
            im = im + gaussian_noise
        return im

    def add_uniform_noise(self,im):
        if random.random() < 0.5:
            print('add_uniform_noise')
            uniform_noise = im.copy()
            cv2.randu(uniform_noise, 0, 1)
            im = im + uniform_noise
        return im

    def randomBlur(self,im):
        if random.random()<0.5:
            print('randomBlur')
            im = cv2.blur(im,(5,5))
        return im

    def randomScale(self,im,boxes):
        if random.random() < 0.5:
            print('randomScale')
            scale = random.uniform(0.8,1.2)
            height,width,c = im.shape
            im = cv2.resize(im,(int(width*scale),height))
            scale_tensor = torch.FloatTensor([[scale,1,scale,1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
        return im,boxes

    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            print('random_bright')
            im = im * alpha + random.randrange(-delta,delta)
            im = im.clip(min=0,max=255).astype(np.uint8)
        return im

    def subMean(self,img,mean):
        mean = np.array(mean, dtype=np.float32)
        img = img - mean
        return img
#########################################parameters #########################################

path = 'C:/Users/dorim/Desktop/computer_vision/final_project'
data,labels,filanames = Buses_data_opening(path)
dataset = BusDataset(root = path,train =True,transform = None,)

def main():
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms

    train_dataset = BusDataset(root=path, train=True, transform=[transforms.ToTensor()])
    train_loader = DataLoader(train_dataset,batch_size=1,shuffle=False,num_workers=0)
    train_iter = iter(train_loader)
    for i in range(100):
        img,target = next(train_iter)
        print(img,target)

if __name__ == '__main__':
    main()



