import torch
import logging
import helper

# Create a logger object
logger = logging.getLogger()

# Set the logger level to INFO
logger.setLevel(logging.INFO)

# Add a handler to direct log output to the console
logger.handlers = [logging.StreamHandler()]

# List of model backbones to train
backbones = ["efficientnet_b0", "resnet26d", "mobilenetv3_small_100"]

# Storing results of trained models
result = {}

for backbone in backbones:
    
    # Logging
    logging.info(f"Training {backbone}")
    
    # Train the model using the current backbone
    result[backbone] = helper.train(backbone)
    
    # Save the training results to a CSV file
    result[backbone].to_csv(f"logs/{backbone}.csv", index=False)