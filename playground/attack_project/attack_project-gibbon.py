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

# device
device = 'cuda:0' if is_available() else 'cpu'

"""
GLOBAL PARAMS
"""

class ProjectAttack:
    def __init__(
        self,
        vm,
        image_size: List[int],
        l2_threshold=7.5,
        steps = 7,
        eps = 8.0/255.0,
        eps_iter = 2.0/255.0,
    ):
        self.vm = vm
        self.image_size = image_size
        self.l2_threshold = l2_threshold
        self.steps = steps
        self.eps = eps
        self.eps_iter = eps_iter

    def attack(self, original, labels, targets):
        """
        PGD Implementation
        """

        # grab hyperparameters from class
        eps = self.eps
        eps_iter = self.eps_iter
        steps = self.steps
        
        # idk they did this in the FGSM code
        targets = [targets]*len(labels)

        # turn original image into a float tensor
        original = torch.from_numpy(original).to(device)

        # init the adversarial image
        adv_img = original.clone().detach()

        # add some noise
        adv_img = adv_img + torch.empty_like(adv_img).uniform_(-eps, eps)
        adv_img = torch.clamp(adv_img, min=0, max=1).detach()


        for step in range(steps):

            # get the grad loss wrt input
            adv_img = adv_img.numpy()
            data_grad = self.vm.get_batch_input_gradient(adv_img, targets) 

            # convert to pytorch tensors
            data_grad = torch.FloatTensor(-data_grad).to(device)
            adv_img = torch.FloatTensor(adv_img).to(device)

            # determine sign of gradient
            grad_sign = data_grad.sign()

            # move the adversarial image in the direction of delta
            adv_img = adv_img.detach() + eps_iter * grad_sign
            delta = torch.clamp(adv_img - original, min=-eps, max=eps)
            adv_img = torch.clamp(original + delta, min=0, max=1)
            adv_img = adv_img.detach()

        # return as a numpy array
        return adv_img.numpy()

