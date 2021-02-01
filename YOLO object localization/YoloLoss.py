import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
 

class YoloLoss(nn.Module):
    def __init__(self,S,B,l_coord,l_noobj):
        super(YoloLoss,self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj
 
    def compute_iou(self, box1, box2):                                                                                                                                                             
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)
 
        lt = torch.max(
            box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )   
 
        rb = torch.min(
            box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )   
 
        wh = rb - lt  # [N,M,2]
        wh[wh<0] = 0  # clip at 0
        inter = wh[:,:,0] * wh[:,:,1]  # [N,M]
 
        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]
 
        iou = inter / (area1 + area2 - inter)
        return iou 
    
    def get_class_prediction_loss(self, classes_pred, classes_target):
        """ 
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)                                                                                                                                        
        classes_target : (tensor) size (batch_size, S, S, 20)
         
        Returns:
        class_loss : scalar
        """
        
        class_loss = F.mse_loss(classes_pred,classes_target,reduction='sum')
        
        return class_loss
         
         
    def get_regression_loss(self, box_pred_response, box_target_response):
        """
        Parameters:
        box_pred_response : (tensor) size (-1, 5)
        box_target_response : (tensor) size (-1, 5)
        Note : -1 corresponds to ravels the tensor into the dimension specified 
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as
         
        Returns:
        reg_loss : scalar
         
        """
         
        reg_loss_xy = F.mse_loss(box_pred_response[:,:2],box_target_response[:,:2],reduction='sum')
        reg_loss_wh =  F.mse_loss(torch.sqrt(box_pred_response[:,2:4]),torch.sqrt(box_target_response[:,2:4]),reduction='sum')
        reg_loss = reg_loss_xy + reg_loss_wh
        
        return reg_loss
         
    def get_contain_object_loss(self, box_pred_response, box_target_response_iou):
        """
        Parameters:
        box_pred_response : (tensor) size ( -1 , 5)
        box_target_response_iou : (tensor) size ( -1 , 5)
        Note : -1 corresponds to ravels the tensor into the dimension specified 
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as
         
        Returns:
        contain_loss : scalar
         
        """
         
        contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response_iou[:,4],reduction='sum')

        return contain_loss
         
    def get_no_object_loss(self, target_tensor, pred_tensor, no_object_mask):
        """                                                                                                                                                                                        
        Parameters:
        target_tensor : (tensor) size (batch_size, S , S, 30)
        pred_tensor : (tensor) size (batch_size, S , S, 30)
        no_object_mask : (tensor) size (batch_size, S , S)
         
        Returns:
        no_object_loss : scalar
         
        """
        no_object_prediction = pred_tensor[no_object_mask].view(-1,30) 
        no_object_target = target_tensor[no_object_mask].view(-1,30)   
        
        no_object_prediction_mask = torch.ByteTensor(no_object_prediction.size()).to(device) 
        no_object_prediction_mask.zero_()
        
        no_object_prediction_mask[:,4]=1 #setting both confidences to 1 
        no_object_prediction_mask[:,9]=1 #setting both confidences to 1 
        
        no_object_prediction_confidence = no_object_prediction[no_object_prediction_mask] 
        no_object_target_confidence = no_object_target[no_object_prediction_mask]         #
        
        no_object_loss = F.mse_loss(no_object_prediction_confidence,no_object_target_confidence,reduction='sum')            
        

        return no_object_loss
         
         
         
    def find_best_iou_boxes(self, bounding_box_target, bounding_box_pred):
        """
        Parameters: 
        box_target : (tensor)  size (-1, 5)
        box_pred : (tensor) size (-1, 5)
        Note : -1 corresponds to ravels the tensor into the dimension specified 
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as
         
        Returns: 
        box_target_iou: (tensor)
        contains_object_response_mask : (tensor)
         
        """
         
        contains_object_response_mask = torch.ByteTensor(bounding_box_target.size()).to(device)
        contains_object_response_mask.zero_()
        box_target_iou = torch.zeros(bounding_box_target.size()).to(device)
        
        for i in range(0,bounding_box_target.size()[0],2):
            
            box1 = bounding_box_pred[i:i+2]
            box1_cord = Variable(torch.FloatTensor(box1.size()))
            
            box1_cord[:,:2] = box1[:,:2]/self.S -0.5*box1[:,2:4] 
            box1_cord[:,2:4] = box1[:,:2]/self.S +0.5*box1[:,2:4]
            
            box2 = bounding_box_target[i].view(-1,5)
            box2_cord = Variable(torch.FloatTensor(box2.size()))
            
            box2_cord[:,:2] = box2[:,:2]/self.S -0.5*box2[:,2:4]
            box2_cord[:,2:4] = box2[:,:2]/self.S +0.5*box2[:,2:4]
            
            iou = self.compute_iou(box1_cord[:,:4],box2_cord[:,:4]) 
            max_iou,max_index = iou.max(0)
            max_index = max_index.data.to(device)
            
            contains_object_response_mask[i+max_index]=1
            box_target_iou[i+max_index,torch.LongTensor([4]).to(device)] = (max_iou).data.to(device)
            
        box_target_iou = Variable(box_target_iou).to(device)

        return box_target_iou, contains_object_response_mask
        

         
    def forward(self, pred_tensor,target_tensor):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30)
                      where B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes
         
        target_tensor: (tensor) size(batchsize,S,S,30)
         
        Returns:
        Total Loss
        '''
        
        N = pred_tensor.size(0)
         
        total_loss = None
        
        contains_object_mask = target_tensor[:,:,:,4] > 0
        no_object_mask = target_tensor[:,:,:,4] == 0
        contains_object_mask = contains_object_mask.unsqueeze(-1).expand_as(target_tensor) #fixing dimensionalities
        no_object_mask = no_object_mask.unsqueeze(-1).expand_as(target_tensor)             #fixing dimensionalities
         
        contains_object_pred = pred_tensor[contains_object_mask].view(-1,30)
        bounding_box_pred = contains_object_pred[:,:10].contiguous().view(-1,5)
        classes_pred = contains_object_pred[:,10:]
                        
        contains_object_target = target_tensor[contains_object_mask].view(-1,30)
        bounding_box_target = contains_object_target[:,:10].contiguous().view(-1,5)
        classes_target = contains_object_target[:,10:]
    

        loss_no_object = self.get_no_object_loss(target_tensor, pred_tensor, no_object_mask)
        
        # Compute the iou's of all bounding boxes and the mask for which bounding box 
        # of 2 has the maximum iou the bounding boxes for each grid cell of each image.
        
        box_target_iou, contains_object_response_mask = self.find_best_iou_boxes(bounding_box_target, bounding_box_pred)
         
        # Create 3 tensors :
        # 1) box_prediction_response - bounding box predictions for each grid cell which has the maximum iou
        # 2) box_target_response_iou - bounding box target ious for each grid cell which has the maximum iou
        # 3) box_target_response -  bounding box targets for each grid cell which has the maximum iou   
        
        box_prediction_response = bounding_box_pred[contains_object_response_mask].view(-1,5)
        box_target_response_iou = box_target_iou[contains_object_response_mask].view(-1,5)
        box_target_response = bounding_box_target[contains_object_response_mask].view(-1,5)    

        # Find the class_loss, containing object loss and regression loss
        
        containing_object_loss = self.get_contain_object_loss(box_prediction_response, box_target_response_iou)
        
        regression_loss = self.get_regression_loss(box_prediction_response, box_target_response)
        
        class_loss = self.get_class_prediction_loss(classes_pred, classes_target)

        

        total_loss = self.l_coord*regression_loss + containing_object_loss + self.l_noobj*loss_no_object + class_loss

        return total_loss / N