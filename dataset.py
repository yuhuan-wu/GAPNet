import torch
import cv2
import os.path as osp
import torch.utils.data
import numpy as np

class Dataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''
    def __init__(self, data_dir, dataset, transform=None, process_label=True, ignore_index=False):
        self.transform = transform
        self.process_label = process_label
        self.ignore_index = ignore_index
        self.img_list = list()
        self.msk_list = list()
        with open(data_dir + dataset + '.txt', 'r') as lines:
            for line in lines:
                line_arr = line.split()
                self.img_list.append(data_dir + line_arr[0].strip())
                self.msk_list.append(data_dir +  line_arr[1].strip())

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_list[idx])
        label = cv2.imread(self.msk_list[idx], 0)
        # print(self.transform)
        if self.transform:
            #try:
            [image, label] = self.transform(image, label)
            #except AttributeError:
            #    print(self.img_list[idx], self.msk_list[idx])

        # Process label if required (similar to gt2gt_ms function)
        if self.process_label:
            label = self.gt2gt_ms(label)
            
        # print("dataset", image.shape)
        return image, label

    def get_img_info(self, idx):
        image = cv2.imread(self.img_list[idx])
        return {"height": image.shape[0], "width": image.shape[1]}
        
    def gt2gt_ms(self, label_tensor):
        """
        Process a single label tensor (1, H, W) to generate multi-scale labels
        Similar to gt2gt_ms in train.py but for a single image
        
        Args:
            label_tensor: Tensor of shape (1, H, W) or (H, W)
            
        Returns:
            Tensor of shape (6, H, W) containing different representations of the label
        """
        # Convert tensor to numpy for OpenCV processing
        if isinstance(label_tensor, torch.Tensor):
            label_np = label_tensor.numpy().astype(np.uint8)
        else:
            label_np = label_tensor.astype(np.uint8)
            
        # Ensure label is 2D (H, W)
        if len(label_np.shape) == 3 and label_np.shape[0] == 1:
            label_np = label_np[0]
            
        # Skip processing if no foreground pixels
        if np.max(label_np) == 0:
            # Return a tensor with all zeros except the first channel which is the original label
            result = np.zeros((6, *label_np.shape), dtype=np.uint8)
            result[0] = label_np
            return torch.from_numpy(result)
            
        # Compute distance transform (same as in gt2gt_ms)
        gt_raw_body = cv2.distanceTransform(label_np, distanceType=cv2.DIST_L2, maskSize=5)
        
        # Get non-zero distance values
        temp = gt_raw_body.flatten()
        temp = temp[temp.nonzero()]
        
        # Process only if we have enough foreground pixels
        if len(temp) > 0:
            # Get threshold for top 20% of distance values
            top30value = temp[np.argpartition(temp, kth=int(np.size(temp)*0.8))[int(np.size(temp)*0.8)]]
            
            if top30value >= 5:
                # Create center, edge, and other regions
                gt_center = gt_raw_body > top30value
                gt_edge = (gt_raw_body < 5) * (gt_raw_body > 0)
                gt_other = label_np - np.logical_or(gt_center, gt_edge)
                
                # Create combined regions with ignore index if needed
                ratio = 255 if self.ignore_index else 0
                gt_edge_other = np.add(np.logical_or(gt_edge, gt_other).astype(int), ratio*gt_center.astype(int))
                gt_center_other = np.add(np.logical_or(gt_center, gt_other).astype(int), ratio*gt_edge.astype(int))
                
                # Stack all representations
                result = np.stack((
                    label_np,                # Original label
                    gt_edge.astype(int),     # Edge region
                    gt_center.astype(int),   # Center region
                    gt_center_other,         # Center + other with edge as ignore index
                    gt_edge_other,           # Edge + other with center as ignore index
                    gt_other.astype(int)     # Other region
                ))
                
                return torch.from_numpy(result)
            
        # If we can't process properly, return a tensor with the original label in first channel
        result = np.zeros((6, *label_np.shape), dtype=np.uint8)
        result[0] = label_np
        return torch.from_numpy(result)
