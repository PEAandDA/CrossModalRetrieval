import json
import os
import pathlib
import random
from collections import defaultdict, Counter

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from .files_folders_tools import delete_folder_contents


def load_dataset_yaml(path):
    """
    Loads a dataset in YAML format from the provided path

    Args:
        path (str): The path to the YAML file

    Returns:
        dict: The infos of the dataset

    Raises:
        FileNotFoundError: If a path within the dataset does not exist, raises a File Not Found error
    """
    try:
        # logging the progress
        print(f"Loading dataset config from {path}...")

        with open(path, 'r', encoding='utf-8') as stream:
            data = yaml.safe_load(stream)

        # Check if the paths in the dataset exist
        for key in ['images', 'captions']:
            data[key] = os.path.abspath(data[key])  # Convert the path to absolute
            if not os.path.exists(data[key]):
                raise FileNotFoundError(f"Path '{data[key]}' does not exist.")

        return data

    except FileNotFoundError as e:
        # Handle the exception here or re-raise it
        raise e


def preprocess_dataset(data):
    """
    Preprocess the dataset function

    Parameters:
    data (dict): Dictionary containing dataset information

    Returns:
    None, generate four json file: train_data.json, val_data.json, test_data.json, vocab.json

    """
    print("Preprocessing dataset...")

    data["output"] = os.path.abspath(data["output"])  # Convert output path to absolute path
    if os.path.exists(data["output"]):  # If the output path exists
        if data["override"]:  # If override is allowed
            delete_folder_contents(data["output"])  # Delete all folder contents in the output path
            os.mkdir(data["output"])  # Create the output directory
        else:
            # Initialize a flag indicating whether any json files are missing
            flag = False

            # Iterate through the four specified json files, constructing their paths
            for path in [os.path.join(data["output"], data["prefix"] + file) for file in
                         ["train.json", "val.json", "test.json", "vocab.json"]]:
                # If a json file does not exist, set the flag to True
                if not os.path.exists(path):
                    flag = True

            # If all json files exist (i.e., the flag is still False)
            if not flag:
                print("All json files are existed.")

                # Assign the constructed paths to respective keys in the 'data' dictionary
                data["vocab"] = f"{data['prefix']}vocab.json"
                data["train"] = f"{data['prefix']}train.json"
                data["val"] = f"{data['prefix']}val.json"
                data["test"] = f"{data['prefix']}test.json"

                # Return from the function
                return

    else:
        os.mkdir(data["output"])  # Create the output directory

    if pathlib.Path(data["captions"]).suffix != ".json":  # If the file suffix is not .json
        raise FileNotFoundError("Captions file must be in json format")  # Raise FileNotFoundError

    else:
        with open(data["captions"], 'r', encoding='utf-8') as stream:  # Open the file stream
            karpathy_json = json.load(stream)  # Load the JSON file

        image_paths = defaultdict(list)  # Create an empty dictionary for image paths
        image_captions = defaultdict(list)  # Create an empty dictionary for image captions
        vocab = Counter()  # Create an empty counter for vocabulary

        for image in karpathy_json["images"]:  # Iterate through each image in the JSON file
            split = image["split"]  # Get the split attribute of the image
            captions = []  # Create an empty list for captions
            for caption in image["sentences"]:  # Iterate through each sentence in the image
                if split != "test":  # If the split is not "test"
                    vocab.update(caption["tokens"])  # Update the vocab counter

                # If the number of tokens in the sentence is less than or equal to max_len
                if len(caption["tokens"]) <= data["max_len"]:
                    captions.append(caption["tokens"])  # Add the token list to the captions list

            if len(captions) == 0:  # If the captions list is empty
                continue  # Continue to the next iteration

            path = os.path.join(data["images"], image["filename"])  # Construct the image path

            image_paths[split].append(path)  # Add the image path to the corresponding split list in image_paths
            image_captions[split].append(
                captions)  # Add the captions list to the corresponding split list in image_captions

        # Get words with count greater than or equal to min_word_count
        words = [w for w in vocab.keys() if vocab[w] >= data["min_word_count"]]
        vocab = {k: v + 1 for v, k in enumerate(words)}  # Update the vocab dictionary, incrementing counts by one

        vocab["<pad>"] = 0  # Set the count for <pad>
        vocab["<unk>"] = len(vocab)  # Set the count for <unk>
        vocab["<start>"] = len(vocab)  # Set the count for <start>
        vocab["<end>"] = len(vocab)  # Set the count for <end>

        data["vocab"] = f"{data['prefix']}vocab.json"
        with open(os.path.join(data["output"], data["vocab"]), 'w',
                  encoding='utf-8') as stream:  # Open the vocab.json file stream
            json.dump(vocab, stream)  # Write the vocab dictionary into the file

        for split in image_paths:  # Iterate through each split in image_paths
            enc_captions = []  # Create an empty list for encoded captions

            for i, path in enumerate(image_paths[split]):  # Iterate through each image path in image_paths[split]

                img = Image.open(path)  # Open the image to confirm the image is valid

                if len(image_captions[split][i]) < data[
                    "captions_num"]:  # If the length of image_captions[split][i] is less than captions_num
                    filled_num = data["captions_num"] - len(
                        image_captions[split][i])  # Calculate the number of captions to fill
                    captions = image_captions[split][i] + [random.choice(image_captions[split][i]) for _ in
                                                           range(filled_num)]  # Fill captions
                else:  # shuffle captions
                    captions = random.sample(image_captions[split][i], data["captions_num"])  # Randomly select captions

                # Assert that the length of captions equals captions_num
                assert len(captions) == data["captions_num"], "Captions length is not equal to captions_num"

                for j, ccaption in enumerate(captions):  # Iterate through each caption in captions
                    enc_caption = [vocab['<start>']] + [vocab.get(word, vocab['<unk>']) for word in ccaption] + [
                        vocab['<end>']]  # Build encoded caption
                    enc_captions.append(enc_caption)  # Add the encoded caption to the enc_captions list

            # Assert that the number of captions equals the product of images and captions_per_image
            assert len(image_paths[split]) * data["captions_num"] == len(enc_captions), \
                "Number of captions is not equal to number of images * captions_per_image"

            data_json = {"images": image_paths[split], "captions": enc_captions}  # Build the data_json dictionary

            # Save the file stream
            data[split] = f"{data['prefix']}{split}.json"
            with open(os.path.join(data["output"], data[split]), 'w', encoding='utf-8') as stream:
                json.dump(data_json, stream)  # Write the data_json dictionary into the file


class ImageTextDataset(Dataset):
    """
    A class representing an Image-Text Dataset.

    Args:
    - dataset_json: Path to the dataset JSON file, must match the value of mode.
    - vocab_json: Path to the vocabulary JSON file.
    - mode: The mode of the dataset, must be "train", "val", or "test".
    - captions_num: Number of captions per image (default: 5).
    - max_len: Maximum length of a caption (default: 30).
    - transform: A transformation function for images (default: None).

    Attributes:
    - mode: The mode of the dataset.
    - captions_num: Number of captions per image.
    - max_len: Maximum length of a caption.
    - data: Dictionary containing image and caption information.
    - vocab: Vocabulary dictionary.
    - dataset_size: Size of the dataset.
    """

    def __init__(self, dataset_json, vocab_json, mode, captions_num=5, max_len=30, transform=None):
        super().__init__()

        self.mode = mode
        assert self.mode in ["train", "val", "test"], "mode must be train, val, or test"

        self.captions_num = captions_num
        self.max_len = max_len

        self.data, self.vocab = self.load_data(dataset_json, vocab_json)
        self.transform = transform

        self.dataset_size = len(self.data["captions"])

    def load_data(self, dataset_json, vocab_json):
        """
        Loads the dataset and vocabulary.

        Args:
        - dataset_json: Path to the dataset JSON file.
        - vocab_json: Path to the vocabulary JSON file.

        Returns:
        - data: Loaded dataset information.
        - vocab: Loaded vocabulary information.
        """

        try:
            with open(dataset_json, 'r', encoding='utf-8') as stream:
                data = json.load(stream)
            with open(vocab_json, 'r', encoding='utf-8') as stream:
                vocab = json.load(stream)
            return data, vocab
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Failed to load data: {e}")

    def __len__(self):
        """
        Returns the size of the dataset.
        """

        return self.dataset_size

    def __getitem__(self, idx):
        """
        Retrieves data at the specified index.

        Args:
        - idx: Index of the data.

        Returns:
        - image: Loaded image data.
        - caption: Loaded caption data.
        """

        image = self.load_image(idx)
        caption = self.load_caption(idx)
        return image, caption

    def load_image(self, idx):
        """
        Loads the image data.

        Args:
        - idx: Index of the data.

        Returns:
        - image: Loaded image data.
        """

        image_idx = idx // self.captions_num
        image = Image.open(self.data["images"][image_idx]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def load_caption(self, idx):
        """
        Loads the caption data.

        Args:
        - idx: Index of the data.

        Returns:
        - caption: Loaded caption data as a padded tensor.
        """

        caption_len = len(self.data["captions"][idx])
        pad_caption = [self.vocab['<pad>']] * (self.max_len + 2 - caption_len)
        caption = torch.LongTensor(self.data["captions"][idx] + pad_caption)
        return caption


def create_dataloader(data, batch_size, workers=4):
    """
    Creates a data loader function.

    Parameters:
    data: A dictionary containing data path and dataset information.
    batch_size: The batch size for processing data.
    workers: The number of worker threads for the data loader, default is 4.

    Returns:
    train_loader: Training data loader.
    val_loader: Validation data loader.
    test_loader: Testing data loader.
    """

    # Preprocessing dataset
    preprocess_dataset(data)

    # Transformation function for training data
    train_transform = transforms.Compose([
        transforms.Resize(256),  # Resize images to 256x256
        transforms.RandomCrop(224),  # Randomly crop images to 224x224
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image data
    ])

    # Transformation function for validation data
    val_transform = transforms.Compose([
        transforms.Resize(256),  # Resize images to 256x256
        transforms.CenterCrop(224),  # Center crop images to 224x224
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image data
    ])

    # Training dataset
    train_set = ImageTextDataset(dataset_json=os.path.join(data["output"], data["train"]),  # Path to image dataset
                                 vocab_json=os.path.join(data["output"], data["vocab"]),  # Path to text dataset
                                 mode="train",  # Dataset mode is set to 'train'
                                 captions_num=data["captions_num"],  # Number of text data
                                 max_len=data["max_len"],  # Maximum length of text data
                                 transform=train_transform)  # Transformation function

    # Validation dataset
    val_set = ImageTextDataset(dataset_json=os.path.join(data["output"], data["val"]),  # Path to image dataset
                               vocab_json=os.path.join(data["output"], data["vocab"]),  # Path to text dataset
                               mode="val",  # Dataset mode is set to 'validation'
                               captions_num=data["captions_num"],  # Number of text data
                               max_len=data["max_len"],  # Maximum length of text data
                               transform=val_transform)  # Transformation function

    # Test dataset
    test_set = ImageTextDataset(dataset_json=os.path.join(data["output"], data["test"]),  # Path to image dataset
                                vocab_json=os.path.join(data["output"], data["vocab"]),  # Path to text dataset
                                mode="test",  # Dataset mode is set to 'test'
                                captions_num=data["captions_num"],  # Number of text data
                                max_len=data["max_len"],  # Maximum length of text data
                                transform=val_transform)  # Transformation function

    # Training data loader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)  # Batch size, shuffle data, number of worker threads, use memory pinning

    # Validation data loader
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, drop_last=False)  # Batch size, do not shuffle data, number of worker threads, use memory pinning, do not drop the last incomplete batch

    # Test data loader
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, drop_last=False)  # Batch size, do not shuffle data, number of worker threads, use memory pinning, do not drop the last incomplete batch

    return train_loader, val_loader, test_loader  # Return training data loader, validation data loader, and test data loader


