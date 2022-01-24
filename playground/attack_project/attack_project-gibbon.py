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

"""
HYPERPARAMETERS
"""

target_label = np.array([7]) 

device = 'cuda' if is_available() else 'cpu'

PARAMS = {  "eps": 8.0/255.0, 
            "gamma": 0.5,
            "steps": 10,
            "SCHED": [60, 85],
            "drop": 5,
            "w_reg": 50, 
            "lin": 25
         }

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

    def MLL(self, x, y):
        """
        Maximum Margin Loss.
        """

        B = y.size(0)
        corr = x[range(B),y]

        x_new = x - 1000*torch.eye(10)[y].to(device)
        tar = x[range(B),x_new.argmax(dim=1)]
        loss = tar - corr
        loss = torch.mean(loss)
        
        return loss

    def attack(self, data, 
               target=target_label, 
               eps=PARAMS["eps"], 
               gamma=PARAMS["gamma"], 
               steps=PARAMS["steps"], 
               SCHED=PARAMS["SCHED"], 
               drop=PARAMS["drop"], 
               w_reg=PARAMS["w_reg"], 
               lin=PARAMS["lin"]):
        """
        An implementation of Guided Adversarial Margin Attack (GAMA).
        
        Currently uses the Frank-Wolfe optimization method.
        
        Paramters:
            @data, The original image.
            @target, The label we wish to 'move towards'.
            @eps, The magnitude of noise. 
            @gama, The amount with which the perturbation gets updated.
            @steps, The number of steps to use.
            @SCHED, The scheduled updates for perturbation.
            @drop, The amount with which gama is updated for each sched-step.
            @w_reg, The initial weighting factor for loss. 
            @lin, The number of regularized update steps to take.
        """
        
        # Make a torch variable of the target
        torch_target = torch.from_numpy(target).to(device)
        
        # Convert the data from numpy to a torch tensor
        torch_data = torch.from_numpy(data).to(device)

        # Get the dimensions of the data
        B,C,H,W = torch_data.size() 
        
        # Init some random delta for perturbation
        torch_delta = torch.rand_like(torch_data).to(device)
        torch_delta = eps+torch.sign(torch_delta-0.5)
        torch_delta.requires_grad=True
        
        # add that delta to the original image for reference
        orig_img = torch_data + torch_delta
        torch_original = torch.tensor(data=orig_img, requires_grad=True)

        # set the WREG value (official implementation uses global var)
        # TODO: investigate wether this is necessary for us
        # -------- I think it has to do with random-restarts
        WREG = w_reg

        pbar = tqdm(range(steps), leave=True);
        # begin to apply perturbations 
        for step in pbar:
            pbar.set_description(f"Perturbation Step: {step}") 

            # handle a scheduled update of gamma     
            if step in SCHED:
                gamma /= drop
        
            torch_delta = torch.tensor(data=torch_delta, requires_grad=True)
            
            # reset the gradient of the perturbed image 
            zero_gradients(torch_delta)
            
            # if we are still making regularized steps
            if step < lin:

                # get model predictions for the original & perturbed images
                combined_batch = (torch_original, torch_data+torch_delta)
                combined_batch = torch.cat(combined_batch, 0).detach()
                out_all = self.vm.get_batch_output(combined_batch.numpy(), 
                                                    labels=target)
                
                # seperate and extract logits
                P_out = nn.Softmax(dim=1)(out_all[:B,:])
                Q_out = nn.Softmax(dim=1)(out_all[B:,:])
                
                # evaluate cost
                regularized_distance = WREG*((Q_out - P_out)).sum(1).mean(0)
                mm_loss = self.MLL(Q_out, torch_target)
                cost = mm_loss + regularized_distance
                
                # updated regularization factor
                WREG -= w_reg/lin
            
            # perform a normal max margin loss update
            else:
                out = model(torch_data+torch_delta)
                Q_out = nn.Softmax(dim=1)(out)
                cost = self.MLL(Q_out, torch_target)
            
            # do back-prop on the cost function
            cost.backward()
            
            # set the delta's gradient to be weighted by epsilon 
            torch_delta.grad = torch.sign(torch_delta.grad)*eps
            
            # take a weighted step towards the new delta
            torch_delta = (1-gamma)*torch_delta + gamma*torch_delta.grad
            
            # clamp delta to be within the specified range
            torch_delta.data.clamp_(-eps, eps) 

        # TODO return a value maestro can understand
        return (torch_data+torch_delta) 


def main():

    """
    HYPERPARAMETERS
    """

    BATCH_SIZE = 1


    # prepare the data loaders and the model
    # used when the student needs to debug on the server
    server_url = "http://128.195.151.199:5000"  
    # used when the student needs to debug locally 
    local_url = "http://127.0.0.1:5000"  

    vm = virtual_model(local_url, application_name="Project_Attack")
    dev_data = get_data(application="Project_Attack", data_type="test")

    targeted_dev_data = []
    for instance in dev_data:
        if instance["label"] != target_label:
            targeted_dev_data.append(instance)

    print(f"Length of targeted_dev_data: {len(targeted_dev_data)}")

    data_gen = DataLoader(
        targeted_dev_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=default_data_collator,
    )

    print("started the process")

    all_vals = []
    correct = 0
    adv_examples = []

    print("start attack test")

    method = ProjectAttack(vm, image_size=[1, 3, 32, 32])


    for batch in data_gen:

        # Call GAMA Attack
        labels = batch["labels"]
        data = batch["image"]
        perturbed_data = method.attack(data)

        # TODO minimize the number of calls to get_batch_output
        output = vm.get_batch_output(perturbed_data, 
                                        labels.cpu().detach().numpy())

        final_pred = np.argmax(output[0])

        if final_pred.item() != labels.item():
            correct += 1

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(
            epsilon, correct, len(test_loader), final_acc)
    )


if __name__ == "__main__":
    main()
