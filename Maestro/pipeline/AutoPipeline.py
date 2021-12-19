import os
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.data import DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    EvalPrediction,
    glue_compute_metrics,
)
from transformers.data.data_collator import default_data_collator
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ------------------ LOCAL IMPORTS ---------------------------------
from Maestro.utils import move_to_device, get_embedding
from Maestro.pipeline import VisionPipeline
from Maestro.data import HuggingFaceDataset, get_dataset
from Maestro.models import build_model
# ------------------ END LOCAL IMPORTS ------------------------------


# ---------------------- ATTACK PIPELINE -------------------------------

class AutoPipelineForVision:
    def __init__(self):
        raise EnvironmentError("Use this like the AutoModel from huggingface")

    @classmethod
    def initialize(
        self,
        pipeline_name,
        dataset_name,
        model_name,
        checkpoint_path,
        scenario,
        training_process=None,
        device=0,
        finetune=True,
    ):
        datasets = get_dataset(dataset_name)
        model = build_model(model_name, num_labels=2, max_length=128, device=device)
        self.device = device
        train_dataset = datasets["train"]
        test_dataset = datasets["test"]
        if finetune:
            model = AutoPipelineForVision.fine_tune_on_task(
                AutoPipelineForVision,
                model,
                train_dataset,
                test_dataset,
                checkpoint_path,
            )
        return VisionPipeline(
            scenario,
            train_dataset,
            test_dataset,
            test_dataset,
            model,
            training_process,
            device,
            None,
        )

    def fine_tune_on_task(
        self,
        model,
        train_dataset,
        validation_dataset,
        checkpoint_path,
    ):
        if not checkpoint_path or not os.path.exists(os.path.join(os.getcwd(), checkpoint_path)):
            print("start training")
            model = self.train(model, train_dataset, self.device)
            torch.save(model.state_dict(), checkpoint_path)
        else:
            model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.to(self.device)
        return model

    @classmethod
    def train(self, model, trainset, device, epoches=10):
        model.train()
        print("trainset is here", trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=10)
        dataset_size = len(trainset)
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        optimizer = optim.Adam(model.parameters())
        for epoch in range(epoches):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
            print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / dataset_size))
            running_loss = 0.0
        return model

# ---------------------- END ATTACK PIPELINE ---------------------------
