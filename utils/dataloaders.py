import json
import os
import pathlib
import random
from collections import defaultdict, Counter

import yaml
from PIL import Image

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

    data["output"] = os.path.abspath(data["output"])  # Convert output path to absolute path
    if os.path.exists(data["output"]):  # If the output path exists
        if data["override"]:  # If override is allowed
            delete_folder_contents(data["output"])  # Delete all folder contents in the output path

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

        with open(os.path.join(data["output"], "vocab.json"), 'w',
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
            with open(os.path.join(data["output"], f"{split}.json"), 'w', encoding='utf-8') as stream:
                json.dump(data_json, stream)  # Write the data_json dictionary into the file
