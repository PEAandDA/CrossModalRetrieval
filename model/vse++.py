import torch
import torchvision.models as models
from torch import nn


class ImageEncoder(nn.Module):
    """
        Image encoder module.
    """

    def __init__(self, embed_size, backbone='resnet152', finetune=True):
        """
        Initialize the ImageEncoder module.

        Args:
        - embed_size (int): The dimension of the image encoding.
        - backbone (str): The backbone architecture to use (default: 'resnet152').
        - finetune (bool): Whether to finetune the backbone (default: True).
        """
        super(ImageEncoder, self).__init__()

        if backbone == 'resnet152':
            # Load the ResNet152 model
            net = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
            # Set whether to finetune the parameters
            for param in net.parameters():
                param.requires_grad = finetune
            # Replace the fully connected layer with a new one
            net.fc = nn.Linear(in_features=net.fc.in_features, out_features=embed_size)
            # Initialize the weights of the new fully connected layer
            nn.init.xavier_uniform_(net.fc.weight)

        elif backbone == 'vgg19':
            # Load the VGG19 model
            net = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
            # Set whether to finetune the parameters
            for param in net.parameters():
                param.requires_grad = finetune
            # Replace the last fully connected layer with a new one
            net.classifier[6] = nn.Linear(in_features=net.classifier[6].in_features, out_features=embed_size)
            # Initialize the weights of the new fully connected layer
            nn.init.xavier_uniform_(net.classifier[6].weight)

        else:
            # Raise an error for unsupported backbone architecture
            raise NotImplementedError(f'backbone {backbone} is not supported')

        # Set the network as an attribute of the module
        self.net = net

    def forward(self, x):
        """
        Perform forward pass through the network.

        Args:
            x : torch.Tensor
                Input tensor.

        Returns:
            torch.Tensor
                Normalized output tensor.
        """
        # Pass input through the network
        out = self.net(x)

        # Normalize the output
        out = nn.functional.normalize(out)

        return out


class TextEncoder(nn.Module):
    """
    Text encoder module.
    """

    def __init__(self, embed_size, word_dim, vocab_size, num_layers):
        """
        Initialize the TextEncoder module.

        Args:
        - embed_size (int): The dimension of the embedding.
        - word_dim (int): The dimension of word embeddings.
        - vocab_size (int): The size of the vocabulary.
        - num_layers (int): The number of recurrent layers.
        """
        super(TextEncoder, self).__init__()

        # Define the word embedding layer
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=word_dim)

        # Define the GRU layer
        self.rnn = nn.GRU(input_size=word_dim, hidden_size=embed_size, num_layers=num_layers, batch_first=True)

        # Initialize the weights of the word embedding layer
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """
        Perform a forward pass through the TextEncoder module.

        Args:
        - x : torch.Tensor
            Input tensor.
        - lengths : torch.Tensor
            Lengths of input sequences.

        Returns:
        - torch.Tensor
            Encoded output tensor.
        """
        # Apply embedding to input tensor
        x = self.embed(x)
        # Pack the sequence for padded input
        packed = nn.utils.rnn.pack_padded_sequence(input=x, lengths=lengths, batch_first=True)
        # Apply the recurrent neural network to the packed sequence
        _, hidden = self.rnn(packed)
        # Normalize and return the last hidden state
        out = nn.functional.normalize(hidden[-1])
        return out


class VSEPP(nn.Module):
    """
    Visual-Semantic Embedding with Perceptual Loss (VSE++)
    """

    def __init__(self, vocab_size, embed_size, word_dim, num_layers,
                 image_backbone='resnet152', finetune=True):
        """
        Initialize VSEPP model.

        Args:
        - vocab_size (int): Size of the vocabulary.
        - embed_size (int): Size of the embedding.
        - word_dim (int): Dimension of word.
        - num_layers (int): Number of layers.
        - image_backbone (str): Type of image backbone.
        - finetune (bool): Whether to finetune the model.
        """
        super(VSEPP, self).__init__()
        # Initialize image encoder
        self.image_encoder = ImageEncoder(embed_size=embed_size,
                                          backbone=image_backbone,
                                          finetune=finetune)
        # Initialize text encoder
        self.text_encoder = TextEncoder(embed_size=embed_size,
                                        word_dim=word_dim,
                                        vocab_size=vocab_size,
                                        num_layers=num_layers)

    def forward(self, images, captions, cap_lens):
        """
        Forward pass of the model.

        Args:
        images: Input images
        captions: Input captions
        cap_lens: Length of the captions

        Returns:
        images_code: Encoded images
        captions_code: Encoded captions
        """

        # Sort the caption lengths and indices
        sorted_cap_len, sorted_cap_indices = torch.sort(cap_lens, 0, descending=True)

        # Reorder the images, captions, and caption lengths based on the sorted indices
        images = images[sorted_cap_indices]
        captions = captions[sorted_cap_indices]
        cap_lens = sorted_cap_len

        # Encode the images and captions
        images_code = self.image_encoder(images)
        captions_code = self.text_encoder(captions, cap_lens)

        # If not in training mode, reorder the encoded images and captions back to their original order
        if not self.training:
            _, recover_indices = torch.sort(sorted_cap_indices)
            images_code = images_code[recover_indices]
            captions_code = captions_code[recover_indices]

        return images_code, captions_code


class TripletLoss(nn.Module):
    """
    Compute the triplet loss
    """

    def __init__(self, margin=0.2, hard_negative=False):
        """
        Initialize the triplet loss module.

        Args:
            margin (float): Margin for triplet loss. Defaults to 0.2.
            hard_negative (bool): Whether to use hard negative mining.
                                  Defaults to False.
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.hard_negative = hard_negative

    def forward(self, images_code, texts_code):
        """
        Compute the triplet loss for given image and text embeddings.

        Args:
            images_code (Tensor): Encoded images.
            texts_code (Tensor): Encoded texts.

        Returns:
            Tensor: The computed loss.
        """
        # Compute similarity matrix between images and texts
        scores = images_code.mm(texts_code.t())

        # Extract diagonal elements and expand for
        # broadcasting in subsequent operations
        diagonal = scores.diag().view(images_code.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # Calculate triplet loss components
        cost_image = (self.margin + scores - d1).clamp(min=0)
        cost_text = (self.margin + scores - d2).clamp(min=0)

        # Create a mask to zero out diagonal elements
        mask = torch.eye(scores.size(0), dtype=torch.bool)
        I = torch.autograd.Variable(mask)

        # Use GPU if available
        if torch.cuda.is_available():
            I = I.cuda()

        # Apply mask to the loss components
        cost_image = cost_image.masked_fill_(I, 0)
        cost_text = cost_text.masked_fill_(I, 0)

        # Perform hard negative mining if enabled
        if self.hard_negative:
            cost_image = cost_image.max(1)[0]
            cost_text = cost_text.max(0)[0]

        # Sum up the loss components
        cost = cost_image.sum() + cost_text.sum()

        return cost



