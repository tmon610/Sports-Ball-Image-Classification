import json
import os

import helper
import torch
from torchvision import datasets
from torchvision import transforms as T
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(2022)

# List of model backbones to train
backbones = ["efficientnet_b0", "resnet26d", "mobilenetv3_small_100"]

# Transformations to apply during pre-processing
transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

#
device = helper.prepare_device()

# Define criterion in evaluation mode
criterion = torch.nn.CrossEntropyLoss().to(device).eval()

# Store evaluation results
result = {}

# List dataset dir
datalist = os.listdir("dataset")

# Tracking progress of runs across backbones
with tqdm(total=len(backbones) * len(datalist)) as pbar:
    for b in backbones:
        
        # Store evaluation results for every backbone
        result[b] = {}
        
        # Create backbone architecture base model
        backbone = helper.pretrained(b)
        
        # Generate the classifier as per configurations used during training
        classifier = helper.generate_classifier(backbone, 15)
        
        # Load the pretrained model weight file
        classifier.load_state_dict(torch.load(f"weights/{b}.pt"))
        
        # Set Evaluation mode
        classifier = classifier.to(device).eval()
        
        # Evaluate on test dataset
        dataset = datasets.ImageFolder("dataset/test", transform=transform)
        
        # DataLoader on test data 
        dataloader = {"test": torch.utils.data.DataLoader(dataset, 512)}
        
        # Store evaluation results of unperturbed data in "default" index
        result[b]["default"] = helper.evaluate(
            dataloader, classifier, criterion, "test"
        )
        
        # Track progress
        pbar.set_postfix(backbone=b, **result[b]["default"])
        
        # Iterate over perturbations while ignoring train, valid and unperturbed test data
        for p in datalist:
            if p in ["train", "valid", "test"]:
                pbar.update()
                continue
            
            # Store result for a particular backbone's perturbation
            result[b][p] = {}
            
            # Generate path of perturbation data location
            path = os.path.join("dataset", p)
            
            # Iterate over images to evaluate perturbations
            for v in os.listdir(path):
                
                # Get dataset and dataloader for perturbed images
                datadir = os.path.join(path, v)
                dataset = datasets.ImageFolder(datadir, transform=transform)
                dataloader = {"test": torch.utils.data.DataLoader(dataset, 512)}
                
                # Evaluate on perturbed images
                result[b][p][v] = helper.evaluate(
                    dataloader, classifier, criterion, "test"
                )
                
                # Track progress
                pbar.set_postfix(backbone=b, perturbation=p, value=v, **result[b][p][v])
                
            # Update progress bar
            pbar.update()
            
        # Clean up variables
        del backbone, classifier

# Store final results in evaluation.json
with open("logs/evaluation.json", "w") as f:
    f.write(json.dumps(result))