import os
from collections import defaultdict
from glob import glob
import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torchvision.transforms as T
from albumentations.pytorch import ToTensorV2
from scipy.signal import convolve2d
from sklearn.metrics import accuracy_score, f1_score
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# initializes a 3x3 Gaussian blur kernel and then normalizes it to ensure that the sum of all elements in the kernel equals 1. 
# This normalization step is crucial for preserving the overall brightness of the image when applying the Gaussian blur filter.

GAUSSIAN_3X3_WEIGHT = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
GAUSSIAN_3X3_WEIGHT = np.divide(GAUSSIAN_3X3_WEIGHT, 16)

# loads an image from the specified directory using OpenCV and converts the image from BGR to RGB color format, optionally resizing it to the specified size. Finally, it converts the image to a NumPy array 

def load_image(directory, size=(224, 224)):
    image = cv2.imread(directory)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if not size is None:
        image = cv2.resize(image, size)
    return np.uint16(image)

# saves an image to a specified file path after converting the image from RGB to BGR color format as OpenCV's imwrite function expects images in BGR format

def save_image(image, save_path):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(save_path, image):
        raise ValueError(f"Unable to save to {save_path}!")

# clips the pixel values of an image array to the range [0, 255]

def clip(image):
    return np.uint16(np.clip(image, 0, 255))

# Perform a convolution operation on an image array using a specified convolution kernel (weight). 
# First determine the dimensions of the image array and check if the image is grayscale or multi-channel (RGB).
# If the image is grayscale, apply the convolution operation with the specified convolution kernel and arguments.
# If the image is multi-channel (RGB), it applies the convolution operation independently to each channel.
# The result is clipped to ensure pixel values are within the valid range

def convolve(image, weight):
    args = dict(mode="same", boundary="symm")
    size = np.shape(image)
    if len(size) < 3:
        image = convolve2d(image, weight, **args)
        return clip(image)
    for i in range(size[-1]):
        image[..., i] = convolve2d(image[..., i], weight, **args)
    return clip(image)

# Apply Gaussian blur to an image by repeatedly convolving it with a Gaussian blur kernel. makes a copy of the input image to avoid modifying the original image. iteratively applies the convolution operation with a predefined Gaussian blur kernel. Finally, it returns the blurred image.

def gaussian_blur(image, num_convolve):
    image_copy = np.copy(image)

    for _ in range(num_convolve):
        image_copy = convolve(image_copy, GAUSSIAN_3X3_WEIGHT)

    return image_copy

# adds Gaussian pixel noise to an image. generates Gaussian noise with a specified standard deviation .  generated noise has the same shape as the input image. adds the noise to the input image and clips the result to ensure pixel values are within the valid range. Finally, it returns the noisy image.

def gaussian_pixel_noise(image, std):
    size = np.shape(image)
    noise = np.random.normal(scale=std, size=size)
    return clip(image + noise)

# scales the contrast of an image by multiplying all pixel values by a scale factor and clips the result to ensure pixel values are within the valid range. returns the contrast-scaled image.

def scale_contrast(image, scale):
    return clip(image * scale)

# changes the brightness of an image by adding a constant value to all pixel values. adds the specified value to the input image and clips the result to ensure pixel values are within the valid range. Finally, it returns the brightness-adjusted image.

def change_brightness(image, value):
    return clip(image + value)

# applies occlusion to an image by randomly selecting a rectangular region of specified edge_length and filling it with zeros. generates random starting points for the occlusion region within the image boundaries. then creates a mask of zeros with the specified edge_length and fills the selected region in the copied image with this mask. Finally, it returns the occluded image after clipping to ensure pixel values are within the valid range.

def occlusion(image, edge_length):
    image_copy = np.copy(image)
    if edge_length > 0:
        size = np.shape(image)

        h_start = np.random.randint(size[0] - edge_length)
        h_end = h_start + edge_length

        w_start = np.random.randint(size[1] - edge_length)
        w_end = w_start + edge_length

        mask = np.zeros([edge_length] * 2).astype(np.int16)
        if len(size) > 2:
            mask = np.expand_dims(mask, -1)

        image_copy[h_start:h_end, w_start:w_end] = mask

    return clip(image_copy)

# adds salt-and-pepper noise to an image by randomly flipping pixel values to 0 (black) or 255 (white) based on a specified rate.  first generates a random mask with values uniformly distributed between 0 and 1. Pixels with values less than rate are selected for noise addition, and pixels with values less than rate * salt_ratio are designated as salt pixels (set to 255). makes a copy of the input image and applies salt-and-pepper noise according to the generated mask. Finally, it returns the noisy image after clipping to ensure pixel values are within the valid range.

def salt_and_pepper(image, rate, salt_ratio=0.5):
    size = np.shape(image)
    mask = np.random.random(size)
    pepper = mask < rate
    salt = mask < rate * salt_ratio

    image_copy = np.copy(image)

    image_copy[pepper] = 0
    image_copy[salt] = 255

    return clip(image_copy)

# prepares device based on availability
def prepare_device():
    
    # Check if CUDA GPU is available and returns it if available
    # Otherwise CPU device is returned
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Return device
    return device


# computes the macro F1-score, which is the average F1-score across all classes. First, it converts the predicted class probabilities (outputs) and true class labels (targets) from PyTorch tensors to NumPy arrays. Then, it selects the predicted class for each sample by finding the index of the maximum probability along the last dimension (class dimension).  Finally, it computes the macro F1-score using f1_score from scikit-learn library with average="macro". The macro F1-score gives equal weight to each class, making it suitable for imbalanced datasets.

def f1_macro(outputs, targets):

    # Convert predicted class probabilities and true class labels to numpy arrays 
    # Get the index of the maximum probability
    outputs = outputs.argmax(-1).cpu().numpy()  
    targets = targets.cpu().numpy()

    #Calculate F1-Score
    return f1_score(targets, outputs, average="macro")

# computes the classification accuracy, which is the fraction of correctly classified samples. first converts the predicted class probabilities (outputs) and true class labels (targets) from PyTorch tensors to NumPy arrays. Then, it selects the predicted class for each sample by finding the index of the maximum probability along the last dimension (class dimension). Finally, it computes the accuracy score. The accuracy score measures the overall correctness of the model's predictions but may be biased towards the majority class in imbalanced datasets.

def accuracy(outputs, targets):

    # Convert predicted class probabilities and true class labels to numpy arrays 
    # Get the index of the maximum probability
    outputs = outputs.argmax(-1).cpu().numpy()
    targets = targets.cpu().numpy()

    #Calculate accuracy
    return accuracy_score(targets, outputs)

# create PyTorch datasets and corresponding data loaders for training, validation, and test sets. 
# Transformations: The function defines a list of transformations to be applied during pre-processing, which include converting images to tensors and normalizing pixel values using mean and standard deviation values [0.485, 0.456, 0.406] and [0.229, 0.224, 0.225], respectively.

# Additional Transformations: Extra transformations are applied on training data.

# Creating Dataset and DataLoader: constructs datasets for training, validation, and test sets using PyTorch's ImageFolder class. t applies the appropriate transformations (transforms_train for training and transforms_eval for validation and test) to the respective datasets. Then, it creates PyTorch DataLoader objects for each dataset, which enable efficient loading of batches of data during training and evaluation. The function returns a dictionary containing the constructed datasets ("train", "valid", "test") and a dictionary containing the corresponding data loaders ("train", "valid", "test").

# Dataloader hyperparameter
def construct_dataset(batch_size):

    # Transformations to apply during pre-processing
    transforms = [
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), #TO-DO recompute for our dataset
    ]

    # Extra Augmentation tranformations on training data
    transforms_train = T.Compose([
        T.RandomRotation(10),      # rotate +/- 10 degrees
        T.RandomHorizontalFlip(),  # reverse 50% of images
        T.Resize(224),             # resize shortest side to 224 pixels
        T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        *transforms])

    # Applying listed transofrmations from 'transforms' list
    transforms_eval = T.Compose(transforms)

    # Creating and then returning dataset dictionary and loaders for training, validation, and test datasets
    dataset = {
        "train": ImageFolder("dataset/train", transform=transforms_train),
        "valid": ImageFolder("dataset/valid", transform=transforms_eval),
        "test": ImageFolder("dataset/test", transform=transforms_eval),
    }
    dataloader = {
        "train": torch.utils.data.DataLoader(
            dataset["train"], batch_size, shuffle=True, pin_memory=True
        ),
        "valid": torch.utils.data.DataLoader(
            dataset["valid"], batch_size, pin_memory=True
        ),
        "test": torch.utils.data.DataLoader(
            dataset["test"], batch_size, pin_memory=True
        ),
    }
    return dataset, dataloader

# used to load a pretrained backbone model and freeze its convolutional layers while retaining the fully connected layers for retraining. 
# Loading Pretrained Backbone: timm.create_model function create the backbone model specified by the backbone argument.

# Freezing Convolutional Layers: function loops through all the parameters of the backbone model. sets requires_grad=False for parameters whose names do not contain "fc". This effectively freezes all convolutional layers of the backbone while retaining the fully connected layers for retraining. Freezing the convolutional layers prevents their weights from being updated during training, which is often desirable when using pretrained models to extract features.

# Overall, this function allows for the convenient loading of pretrained backbone models and customization of their training behavior by selectively freezing certain layers while keeping others trainable. It's commonly used in transfer learning scenarios where pretrained models are adapted to new tasks with limited data.

def pretrained(backbone):
    backbone = timm.create_model(backbone, pretrained=True)
    
    # for p in backbone.parameters():
    #     p.requires_grad = False

     # Freeze the convolutional layers and retain the fully connected layers for retraining
    for name, param in backbone.named_parameters():
        if "fc" not in name:  # Exclude fully connected layers from freezing
            param.requires_grad = False
    
    return backbone

# replace the final fully connected layer of a backbone model with a new linear layer to adapt the model for a specific classification task. Additionally, it allows for the inclusion of a dropout layer with a specified dropout probability. 

# Overall, this function enables the customization of backbone models for classification tasks by replacing the final fully connected layer and optionally adding dropout regularization. It's commonly used in transfer learning scenarios to adapt pretrained models to new classification tasks.

# Defining Dropout Layer: defines a dropout layer based on the provided dropout probability. If prob is greater than 0, it creates a Dropout layer with the specified probability; otherwise effectively creating a layer that does not perform dropout.

# Replacing Fully Connected Layer: Depending on the type of backbone model (backbone), the function identifies the appropriate attribute representing the final fully connected layer and replaces it with a new linear layer followed by the dropout layer. modifies the respective attributes of different backbones by replacing them with a new sequential module containing the dropout layer followed by a linear layer with num_classes output features.
 
def generate_classifier(backbone, num_classes, p_dropout=0):

    # Define dropout layer based on provided dropout probability
    # Default value is 0 in the function. Use the layer only if dropout probability > 0
    dropout = torch.nn.Identity()
    if p_dropout > 0:
        dropout = torch.nn.Dropout(p_dropout)

    # Identify the family of backbone
    # Replace the final fully connected layer of the backbone with a new linear layer
    # Replace dropout layer if p_dropout is defined
        
    if isinstance(backbone, timm.models.resnet.ResNet):
        in_features = backbone.fc.in_features
        backbone.fc = torch.nn.Sequential(
            torch.nn.ReLU(), dropout,
            torch.nn.Linear(in_features, num_classes),
            torch.nn.LogSoftmax(dim=1)
        )
        
    if isinstance(backbone, timm.models.efficientnet.EfficientNet):
        in_features = backbone.classifier.in_features
        backbone.classifier = torch.nn.Sequential(
            torch.nn.ReLU(), dropout,
            torch.nn.Linear(in_features, num_classes),
            torch.nn.LogSoftmax(dim=1)
        )
        
    if isinstance(backbone, timm.models.mobilenetv3.MobileNetV3):
        in_features = backbone.classifier.in_features
        backbone.classifier = torch.nn.Sequential(
            torch.nn.ReLU(), dropout,
            torch.nn.Linear(in_features, num_classes),
            torch.nn.LogSoftmax(dim=1)
        )

    # Return modified backbone
    return backbone


# train a model for one epoch (one pass through the entire training dataset).  

def train_one_epoch(
    dataloader,        #Dictionary containing PyTorch data loaders for different subsets (e.g., train, validation).
    model,             #The PyTorch model to be trained.
    criterion,         #The loss function used for optimization.
    optimizer,         #The optimizer used for updating model parameters.
    scheduler=None,    #(Optional) Learning rate scheduler.
    scaler=None,       #(Optional) PyTorch's GradScaler for automatic mixed precision training.
    ema=None,          #(Optional) Exponential Moving Average (EMA) for model parameter updates.
    subset="train",    #Specifies the subset on which training/validation is performed.
):
    # Dictionary for storage of training stats and define the subset(training/validation)
    record = defaultdict(float)
    record["Subset"] = subset.title()
    
    device = prepare_device()

    # Enable automatic mixed precision training if scaler is provided
    with torch.autocast(device):

        # Set training mode
        model.train()

# Training Loop: function iterates over batches in the specified subset. moves input and target data to the GPU. Performs a forward pass to obtain model predictions and calculate the loss between predictions and targets. Zeroes the gradients and updates model parameters through backpropagation. If using mixed precision training, scales the loss and performs optimizer step and update. Adjusts the learning rate if a scheduler is provided. Updates the Exponential Moving Average (EMA) of the model parameters if provided. Accumulates training statistics such as loss, accuracy, and F1-macro.

        # Iterate over batches in the specified subset
        for inputs, targets in dataloader["train"]:

            # Move input and target data to GPU
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Predict outputs as a forward pass and calculate loss
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Zero gradients and update weights as a backward pass
            optimizer.zero_grad()

            # If using mixed precision training scale the loss
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # Adjust learning rate if scheduler is provided
            if not scheduler is None:
                scheduler.step()

            # Update Exponential Moving Average (EMA) if provided
            if not ema is None:
                ema.update(model)

            # Update training stats for the batch
            record["Loss"] += loss.item()
            record["Accuracy"] += accuracy(outputs, targets)
            record["F1-Macro"] += f1_macro(outputs, targets)

        # After processing all batches in the epoch, it calculates the average loss, accuracy, and F1-macro for the entire subset.
        # Calculate training stats for the entire subset
        record["Loss"] /= len(dataloader["train"])
        record["Accuracy"] /= len(dataloader["train"])
        record["F1-Macro"] /= len(dataloader["train"])

    # Return training stats for this epoch
    return dict(record)


def evaluate(dataloader, model, criterion, subset):

    # Dictionary for storage of evaluation stats and define the subset
    record = defaultdict(float)
    record["Subset"] = subset.title()
    
    device = prepare_device()

    with torch.autocast(device), torch.inference_mode():

        # Set evaluation mode
        model.eval()

        # Iterate over batches in the specified subset
        for inputs, targets in dataloader[subset]:

            # Move input and target data to the appropriate device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Predict outputs as a forward pass and calculate loss
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Update evaluation stats for the batch
            record["Loss"] += loss.item()
            record["Accuracy"] += accuracy(outputs, targets)
            record["F1-Macro"] += f1_macro(outputs, targets)

        # Calculate evaluation stats for the entire subset
        record["Loss"] /= len(dataloader)
        record["Accuracy"] /= len(dataloader)
        record["F1-Macro"] /= len(dataloader)

    # Return evaluation stats for this epoch
    return dict(record)

# trains a classifier model using a specified backbone architecture.

def train(backbone):
    
    torch.manual_seed(2022)
    epochs = 200
    
    device = prepare_device()

    # Construct dataset and dataloader
    # create the dataset and corresponding dataloaders. The input size for images is set to 512x512 pixels
    dataset, dataloader = construct_dataset(512)

    #Determines the number of classes in the dataset.
    num_classes = len(dataset["train"].classes) 

    # Generate a classifier model using the specified backbone and pre-trained weights, 
    # define loss function, optimiser using SGD, Learning Rate scheduler using Cosine Annealing

    # generate_classifier() function to generate a classifier model using the specified backbone architecture. The backbone is pretrained, and its final fully connected layer is replaced with a new linear layer with the number of output classes.  Considering CrossEntropyLoss as the loss function, we initialize the optimizer with stochastic gradient descent (SGD), with a learning rate determined by the batch size and momentum set to 0.9. The learning rate scheduler using Cosine Annealing, which adjusts the learning rate over the course of training.
    
    model = generate_classifier(pretrained(backbone), num_classes).to(device)
    
    # Define criterion for computing the loss
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    # Define optimizer for updating model parameters
    # Using Adam optimizer with a learning rate scaled based on batch size
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001 * dataloader["train"].batch_size / 256,
    )
    
    # Define learning rate scheduler
    # Using Cosine Annealing Learning Rate Scheduler
    # Adjusts the learning rate over the course of training epochs
    # The learning rate will decrease from the initial LR to 1e-6 in a cosine manner
    # The total number of iterations for the scheduler is set to 50 times the number of batches in training data
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 50 * len(dataloader["train"]), 1e-6
    )

    # Initialize a gradient scaler for mixed precision training 
    # and a list to store training history
    
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else torch.cpu.amp.GradScaler()
    history = []

    # Progress bar for visualisation and after every training epoch
    # evaluate model on validation set and then update the progress bar
    with tqdm(total=epochs) as pbar:

        # Iterates through 200 epochs. For each epoch, trains the model for one epoch using the train_one_epoch() function passing the dataset, model, criterion, optimizer, and scheduler. After each epoch, evaluates the model's performance on the validation set using the evaluate() function.
        
        for _ in range(pbar.total):
            history.append(
                train_one_epoch(
                    dataloader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    scaler,
                )
            )

            history.append(evaluate(dataloader, model, criterion, "valid"))

            pbar.set_postfix(history[-1])
            pbar.update()

    # # Saves the trained model's weights to a file in the "weights" directory, delete model to free memory
    torch.save(model.state_dict(), f"weights/{backbone}.pt")
    del model

    # Returns a Pandas DataFrame containing the training history, which includes loss, accuracy, and F1-macro scores for each epoch on both the training and validation sets.
    
    return pd.DataFrame(history)