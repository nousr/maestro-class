import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader
from Maestro.evaluator.Evaluator import get_data
from typing import List, Iterator, Dict, Tuple, Any, Type
from transformers.data.data_collator import default_data_collator
from Maestro.attacker_helper.attacker_request_helper import virtual_model


"""
GLOBAL HYPERPARAMETERS

WREG, the default regularization constant used by GAMA.
SCHED, the defualt schedule used by GAMA.
EPS, the defualt EPS value used by GAMA.
"""

WREG = 50
SCHED = [60, 85]
EPS = 8.0/255.0


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

    def attack(self, data, target, eps, gamma, steps, SCHED, drop, w_reg, lin):
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
        torch_target = torch.Variable(target)
        
        # Convert the data from numpy to a torch tensor
        data = torch.from_numpy(data)

        # Get the dimensions of the data
        B,C,H,W = data.size() 
        
        # Init some random delta for perturbation
        delta = torch.rand_like(data)
        delta = eps+torch.sign(delta-0.5)
        delta.requires_grad=True
        
        # add that delta to the original image for reference
        orig_img = data + delta
        t_orig_img = torch.Variable(orig_img, requires_grad=False)

        # set the WREG value (official implementation uses global var)
        # TODO: investigate wether this is necessary for us
        # -------- I think it has to do with random-restarts
        WREG = w_reg

        # begin to apply perturbations 
        for step in tqdm(range(steps)):
            
            # handle a scheduled update of gamma     
            if step in SCHED:
                gamma /= drop
        
            delta = Variable(delta, requires_grad=True)
            
            # reset the gradient of the perturbed image 
            zero_gradients(delta)
            
            # if we are still making regularized steps
            if step < lin:

                # TODO implement this for maestro 
                # get model predictions for the original & perturbed images
                out_all = model(torch.cat((orig_img, data+delta), 0))
                
                # seperate and extract logits
                P_out = torch.nn.Softmax(dim=1)(out_all[:B,:])
                Q_out = torch.nn.Softmax(dim=1)(out_all[B:,:])
                
                # evaluate cost
                regularized_distance = WREG*((Q_out - P_out)).sum(1).mean(0)
                mm_loss = max_margin_loss(Q_out, tar)
                cost = mm_loss + regularized_distance
                
                # updated regularization factor
                WREG -= w_reg/lin
            
            # perform a normal max margin loss update
            else:
                out = model(data+delta)
                Q_out = nn.Softmax(dim=1)(out)
                cost = max_margin_loss(Q_out, tar)
            
            # do back-prop on the cost function
            cost.backward()
            
            # set the delta's gradient to be weighted by epsilon 
            delta.grad = torch.sign(delta.grad)*eps
            
            # take a weighted step towards the new delta
            delta = (1-gamma)*delta + gamma*delta.grad
            
            # clamp delta to be within the specified range
            delta.data.clamp_(-eps, eps) 

        # TODO return a value maestro can understand
        return (data+delta) 


def main():

    # prepare the data loaders and the model
    # used when the student needs to debug on the server
    server_url = "http://128.195.151.199:5000"  
    # used when the student needs to debug locally 
    local_url = "http://127.0.0.1:5000"  

    vm = virtual_model(local_url, application_name="Project_Attack")
    target_label = 7
    dev_data = get_data(application="Project_Attack", data_type="test")

    targeted_dev_data = []
    for instance in dev_data:
        if instance["label"] != target_label:
            targeted_dev_data.append(instance)

    print(f"Length of targeted_dev_data: {len(targeted_dev_data)}")

    # there was a dumb self-assignment here 
    universal_perturb_batch_size = 1

    iterator_dataloader = DataLoader(
        targeted_dev_data,
        batch_size=universal_perturb_batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
    )

    print("started the process")

    all_vals = []
    correct = 0
    adv_examples = []

    print("start testing")

    # Loop over all examples in test set
    test_loader = iterator_dataloader
    epsilon = 0.214
    method = ProjectAttack(vm, image_size=[1, 3, 32, 32])

    for batch in test_loader:
        # Call GAMA Attack
        labels = batch["labels"]

        perturbed_data = method.attack(
            batch["image"].cpu().detach().numpy(),
            labels.cpu().detach().numpy(),
            [target_label],
            epsilon=epsilon,
        )

        # TODO minimize the number of calls to get_batch_output
        output = vm.get_batch_output(perturbed_data, labels.cpu().detach().numpy(),)
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
