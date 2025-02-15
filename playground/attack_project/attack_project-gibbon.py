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
        steps = 15,
        eps = 7.5/255.0,
        eps_iter = 4.5/255.0,
    ):
        self.vm = vm
        self.image_size = image_size
        self.l2_threshold = l2_threshold
        self.steps = steps
        self.eps = eps
        self.eps_iter = eps_iter

    def attack(self, original, labels, targets):

        """
        PGD Implementation w/ scheduled epsilon

        get an image
        for i in range():
            add a bit of noise
            check check prediction
            if success
                return 
            otherwise
                continue
        """

        # CIFAR10 classes
        all_classes = [0,1,2,3,4,5,6,7,8,9]

        # grab hyperparameters from class
        eps = self.eps
        eps_iter = self.eps_iter
        steps = self.steps
        
        # idk they did this in the FGSM code
        targets = [targets]*len(labels)

        # turn original image into a float tensor
        original = torch.from_numpy(original).to(device)

        for i in range(10):

            # walk the eps up if we're failing
            if i > 0:
                eps += 0.04
                eps_iter += 0.0055
            
            # init the adversarial image
            adv_img = original.clone().detach() 

            # add some noise
            adv_img = adv_img + torch.empty_like(adv_img).uniform_(-eps, eps)
            adv_img = torch.clamp(adv_img, min=0, max=1).detach()

            for step in range(steps):

                # get the grad loss wrt input
                data_grad = self.vm.get_batch_input_gradient(adv_img, targets) 

                # convert to pytorch tensors
                data_grad = torch.FloatTensor(-data_grad).to(device)
                adv_img = torch.FloatTensor(adv_img).to(device)

                # determine sign of gradient
                grad_sign = data_grad.sign()

                # move the adversarial image in the direction of delta
                adv_img = adv_img.detach() + eps_iter * grad_sign

                # finds how much we've changed from the original image
                delta = torch.clamp(adv_img - original, min=-eps, max=eps)
                adv_img = torch.clamp(original + delta, min=0, max=1)

                # prepare to send to virtual machine
                adv_img = adv_img.detach().numpy()

                # get logit values from
                logits = self.vm.get_batch_output(adv_img, all_classes)

                # convert back to pytorch
                logits = torch.tensor(logits).to(device)

                # convert logits to probability dist
                probs = nn.Softmax(dim=1)(logits).detach().numpy()

                # check prediction
                if (np.argmax(probs) == targets[0]):
                    return adv_img

        # if we got here...oof
        return adv_img
