import pickle
import torch
import numpy as np
from PIL import Image
from diffusion_model import DiffusionModel


class ImageGenerator:
    def __init__(self, denoising_steps = 1000, p_uncond = .2):

        #####################################################
        # TODO: Initialize diffusion model params           #
        #                                                   #
        #       Hint: Expects images in (C, H, W) format    #
        #       Hint: condition_dim will ultimetaly define  #
        #             num_embeddings for torch.nn.Embedding #
        #             layer of the 2D noise prediction net. #
        #             How might CFG effect this?            #
        #                                                   #
        #####################################################
        self.input_shape = None
        self.condition_dim = None
        #####################################################
        #                   END OF YOUR CODE                #
        #####################################################

        self.policy = DiffusionModel(
            input_shape = self.input_shape, 
            condition_dim = self.condition_dim, 
            sequential = False, 
            denoising_steps = denoising_steps, 
            p_uncond = p_uncond
            )
    
    def load_dataset(self, dataset_paths, batch_size = 64):

        all_data = []
        all_labels = []

        for dataset_path in dataset_paths:
            with open(dataset_path, 'rb') as file:
                data_dict = pickle.load(file, encoding='bytes')

            data = data_dict[b'data']
            labels = data_dict[b'labels']

            labels = np.array(labels)
            labels = labels + 1

            # Reshape and normalize data between [-1, 1] -> 0 is -1, 255 is 1
            data = data.reshape(-1, 3, 32, 32)
            data = data / 255
            data = data * 2 - 1

            # Only train on the first 3 classes
            mask = np.isin(labels, [1, 2, 3])
            data = data[mask]
            labels = labels[mask]

            all_data.append(data)
            all_labels.append(labels)

        # Concatenate the list of arrays along the first axis to create a single dataset
        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_data = torch.Tensor(all_data).float()
        all_labels = torch.Tensor(all_labels).int()

        ####################################################
        # TODO: Add data augmentation                      #
        #       Make sure to keep the labels the same      #
        #       Don't get rid of the original data         #
        #                                                  #
        #       Two augmentations are requires, feel free  #
        #       to do more if you want to                  #
        ####################################################


        ####################################################
        #                   END OF YOUR CODE               #
        ####################################################



        dataset = torch.utils.data.TensorDataset(all_data, all_labels)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.data_loader = data_loader
    
    def train_policy(self, epochs = 10):
        self.policy.train(train_epochs = epochs, data_loader=self.data_loader)

    def generate_images(self, guidance = 0, num_samples = 10, threshold = False, class_label = None):
        """
        generate an image
        Args:
            num_samples (int): the number of images to generate per class
            guidance (float): the amount of guidance to use
            class_label (int): If None, will generate all classes num_samples times. Otherwise 3 * num_samples images of the given class
        Returns:
            images (np.array): the generated images of shape (3 * num_samples, H, W, C)
                - the first num_samples images are class 1
                - the second num_samples images are class 2
                - the third num_samples images are class 3
        """
        if class_label is None:
            airplanes = [1] * num_samples
            automobiles = [2] * num_samples
            birds = [3] * num_samples

            all_labels = airplanes + automobiles + birds

        else:
            all_labels = [class_label] * num_samples * 3

        # The size of this conditioning tensor is different than what you would expect given the function signature of generate_sample
        # If you are looking at this before implementing generate_sample, the first dimension in this cond tensor is still the batch size
        # The generate_sample function signature reflects the shape of the conditioning tensor for the 1D prediction network (part 2.4)
        # If none of this makes sense to you, that is fine, you do not need to worry about any of this
        cond = torch.tensor(all_labels).to(self.policy.device)

        # generate the image
        image = self.policy.generate_sample(cond, guidance_weight = guidance, threshold=threshold)
        image = (image.clamp(-1, 1) + 1) / 2
        image = image.detach().cpu().numpy()
        image = (image * 255).astype(np.uint8)
        
        # transpose batch from (B, C, H, W) to (B, H, W, C)
        image = np.transpose(image, (0, 2, 3, 1))

        return image






        
    
    
