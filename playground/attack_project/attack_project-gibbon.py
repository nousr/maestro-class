import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
from torch import Tensor
from torch.cuda import is_available
from torch.utils.data import DataLoader
from Maestro.evaluator.Evaluator import get_data
from torch.autograd.gradcheck import zero_gradients
from typing import List, Iterator, Dict, Tuple, Any, Type
from transformers.data.data_collator import default_data_collator
from Maestro.attacker_helper.attacker_request_helper import virtual_model


class ProjectAttack:
    def __init__(
        self,
        vm,
        image_size: List[int],
        l2_threshold=7.5

    ):
        self.vm = vm
        self.image_size = image_size
        self.l2_threshold = l2_threshold


    def attack(self, original_image, labels, target_label, eps=8.0/255.):
        
        targets = [target_label]*len(labels)
    
        # turn original image into a float tensor
        original_image = torch.from_numpy(original_image)

        # init the adversarial image
        adv_img = original_image.clone().detach()
        
        # add some noise
        adv_img = adv_img + torch.empty_like(adv_img).uniform_(-eps, eps)
        adv_img = torch.clamp(adv_img, min=0, max=1).detach()
        
        for _ in range(100):
            
            # get the grad loss wrt input
            adv_img = adv_img.numpy()
            data_grad = self.vm.get_batch_input_gradient(adv_img, targets) 
            data_grad = torch.FloatTensor(data_grad)
            
            adv_img = torch.FloatTensor(adv_img)
            
            # determine sign of gradient
            grad_sign = data_grad.sign()
            
            adv_img = adv_img.detach() + eps * -grad_sign
            delta = torch.clamp(adv_img - original_image, min=-eps, max=eps)
            adv_img = torch.clamp(original_image + delta, min=0, max=1)
            adv_img = adv_img.detach()
        
        return adv_img.numpy()
        
