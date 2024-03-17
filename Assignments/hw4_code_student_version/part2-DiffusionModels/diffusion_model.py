import torch
from randomizer import DL_random
from tqdm import tqdm
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from noise_prediction_net import ConditionalUnet1D
from noise_prediction_net import ConditionalUnet2D
from noise_prediction_net import TesterModel
from noise_scheduler import NoiseScheduler

class DiffusionModel:

    def __init__(self, input_shape, condition_dim, sequential = True, denoising_steps = 100, p_uncond = .2, weights_path = None, test = False):
        """
        Initialize the diffusion model
        Args:
            input_shape (tuple): the shape of the input data
            condition_dim (int): the dimension of the conditioning information
            sequential (bool): whether to use the sequential model (for robotics)
            denoising_steps (int): the number of denoising steps
            p_uncond (float): the probability of unconditional training
            weights_path (str): the path to load the weights from
            test (bool): whether to use the test model (only for unit tests and autograder, do not set this yourself)
        """
        # saves the number of training epochs that have been run
        self.training_epoch = 0
        self.input_shape = input_shape
        self.condition_dim = condition_dim
        self.sequential = sequential
        self.p_uncond = p_uncond
        self.denoising_steps = denoising_steps

        # initialize the noise scheduler with the number of denoising steps
        self.noise_scheduler = NoiseScheduler(self.denoising_steps)

        # initialize the noise prediction network
        if test:
            # Used for unit tests and autograder
            self.noise_pred_net = TesterModel(input_shape, condition_dim)
        elif self.sequential:
            # Used for robotics 
            T, input_dim = input_shape
            self.noise_pred_net = ConditionalUnet1D(
                input_dim = input_dim, 
                global_cond_dim = self.condition_dim
                )
        else:
            # Used for image generation
            c_in, H, W = input_shape
            assert H == W
            self.noise_pred_net = ConditionalUnet2D(
                input_dim = H, 
                c_in = c_in, 
                c_out = c_in,
                global_cond_dim = self.condition_dim
                )
        
        # initialize the device, ema model (helps performance, don't worry about calling this), and optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ema = EMAModel(model=self.noise_pred_net,power=0.75)
        # load weights if they are provided
        if weights_path is not None:
            self.noise_pred_net.load_state_dict(torch.load(weights_path, map_location=self.device))
            ema_path = weights_path.replace(".pth", "_ema.pth")
            self.ema.averaged_model.load_state_dict(torch.load(ema_path, map_location=self.device))
        self.noise_pred_net.to(self.device)
        self.ema.averaged_model.to(self.device)
        self.optimizer = torch.optim.Adam(self.noise_pred_net.parameters(), lr=1e-4, weight_decay=1e-6)
        
        

    def train(self, data_loader, train_epochs = 10):
        """
        train the diffusion model
        Args:
            data_loader (nn.Module): the diffusion model
            train_epochs (torch.utils.data.Dataset): the dataset
        """
        self.noise_pred_net.train()
        self.lr_scheduler = get_scheduler(
                name='cosine',
                optimizer=self.optimizer,
                num_warmup_steps=500,
                num_training_steps=len(data_loader) * train_epochs
                )
        
        with tqdm(range(train_epochs), position = 1, desc = 'Training Progress') as tdqm_epochs:
            for epoch in tdqm_epochs:
                average_loss = 0
                with tqdm(data_loader, position = 0, desc = 'Batch') as tdqm_data_loader:
                    for data, cond in tdqm_data_loader:
                        data = data.to(self.device)
                        cond = cond.to(self.device)
                        loss = self.compute_loss_on_batch(data, cond)
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.lr_scheduler.step()
                        self.ema.step(self.noise_pred_net)
                        average_loss += loss.item()
                
                average_loss /= len(data_loader)
                tdqm_epochs.set_postfix({"Epoch": self.training_epoch, "Loss" : average_loss})
                self.training_epoch += 1

    def save_weights(self, path):
        """
        save the weights of the diffusion model to path
        save the weights of the ema model to path[:-4] + "_ema.pth"
        Args:
            path (str): the path to save the weights to
        """
        torch.save(self.noise_pred_net.state_dict(), path)
        torch.save(self.ema.averaged_model.state_dict(), path[:-4] + "_ema.pth")

    def load_weights(self, path):
        """
        load the weights of the diffusion model from path
        load the weights of the ema model from path[:-4] + "_ema.pth"
        Args:
            path (str): the path to load the weights from
        """
        self.noise_pred_net.load_state_dict(torch.load(path, map_location=self.device))
        self.ema.averaged_model.load_state_dict(torch.load(path[:-4] + "_ema.pth", map_location=self.device))
        self.noise_pred_net.to(self.device)
        self.ema.averaged_model.to(self.device)

    def compute_loss_on_batch(self, data, cond, seed = None):
        """
        ** Use DL_random() to generate any random numbers **

        train the diffusion model
        Args:
            cond (torch.Tensor): the conditioning information of shape (B, self.condition_dim)
            data (torch.Tensor): the data of shape (B, self.input_shape)
        Returns:
            loss (torch.Tensor): the training loss

        """
        #######################################################################
        # TODO: Implement unconditional training for classifier free guidance #
        #       
        # Hint: DL_random(shape = (1,), normal = False, seed = seed).item()   #
        #       returns a random value between 0 and 1                        #
        # Hint: only a couple lines, 1 call to DL_random                      #
        #######################################################################

        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################

        loss = None

        ########################################
        # TODO: Implement loss comptutation    #
        # Hint: 2 calls to DL_random           #
        ########################################

        ########################################
        #             END OF YOUR CODE         #
        ########################################

        return loss

    def generate_sample(self, cond, guidance_weight = 0.0, threshold = False, seed = None):
        """
        ** Use DL_random() to generate any random numbers **

        generate an output from the diffusion model
        Args:
            cond (torch.Tensor): the conditioning information of shape (B, self.condition_dim)
        Returns:
            sample (torch.Tensor): the generated sample of shape (B, self.input_shape)
        """
        noise_pred_net = self.ema.averaged_model
        noise_pred_net.eval()
        sample = None

        with torch.no_grad():
            ###################################################################################
            ##TODO: Generate a sample from the diffusion model using classifier free guidance##
            # Hint: 1 call to DL_random                                                       #
            # Hint: Use noise_pred_net defined above, not self.noise_pred_net                 #
            ###################################################################################
            pass
            ####################################################################################
            #                                 END OF YOUR CODE                                 #
            ####################################################################################
        noise_pred_net.train()
        return sample

