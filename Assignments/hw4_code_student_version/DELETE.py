


import torch
def DL_random(shape, int_range = None, normal = True, seed=None):
    """Generate a random matrix of the given shape.
    If int_range is provided, the matrix will be filled with integers in the range [int_range[0], int_range[1])
    If int_range is not provided, the matrix will be filled with random floats (uniform or normal distribution)

    * Do not modify this function *
    * Seed parameter is for unit tests and autograder *

    Args:
        shape (tuple): the shape of the matrix to generate
        int_range (tuple): the range of integers to use (optional)
        normal (true): whether to use uniform or normal distribution - default is normal
        seed (int): the seed to use for random number generation
    
    Returns:
        torch.Tensor: the generated matrix

    """
    if seed is not None:
        torch.manual_seed(seed)
    if int_range is not None:
        assert len(int_range) == 2
        return torch.randint(*int_range, shape)
    else:
        if normal:
            return torch.randn(*shape)
        else:
            return torch.rand(*shape)







from randomizer import DL_random
import numpy as np
import torch

class NoiseScheduler:

    def __init__(self, num_steps, beta_start = 0.0001, beta_end = 0.02):
        # initialize the beta parameters (variance) of the scheduler
        self.beta_start = beta_start
        self.beta_end = beta_end

        # number of inference steps (same as num training steps)
        self.num_steps = num_steps

        # linear schedule for beta
        self.betas = np.linspace(self.beta_start, self.beta_end, self.num_steps)

        ###########################################################
        # TODO: Compute alphas and alpha_bars (refer to DDPM paper)
        ###########################################################
        self.alphas = 1 - self.betas
        # self.alpha_bars = np.ones_like(self.alphas)
        # for i in range(len(self.alphas)):
        #     self.alpha_bars[i] = np.prod(self.alphas[:i+1])

        self.alpha_bars = np.cumprod(self.alphas) # tip on piazza to replace for loop
        
        ###########################################################
        #                     END OF YOUR CODE                    #
        ###########################################################

        # convert to tensors
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alphas = torch.from_numpy(self.alphas).to(self.device).float()
        self.alpha_bars = torch.from_numpy(self.alpha_bars).to(self.device).float()
        self.betas = torch.from_numpy(self.betas).to(self.device).float()

    def denoise_step(self, model_prediction, t, x_t, threshold = False, seed = None):
        """
        ** Use DL_random() to generate any random numbers **

        Implement a step of the reverse denoising process
        Args:
            model_prediction (torch.Tensor): the output of the noise prediction model (B, input_shape)
            t (int): the current timestep
            x_t (torch.Tensor): the previous timestep (B, input_shape)
            threshold (bool): whether to threshold x_0, implemented in part 2.3
        Returns:
            x_t_prev (torch.Tensor): the denoised previous timestep (B, input_shape)
        
        """

        x_t_prev = None
        if not threshold:
            #####################################
            # TODO: Implement a denoising step  #
            # Hint: 1 call to DL_random         #    
            #####################################

            ## Following Algorithm 2 - Sampling (we need eqn 4). Assumming t is always just one scalar.
            # print('t:', t, 'x_t', x_t.shape)    # x_t is NOT previous timestep, as the docstring says. Instead it is the noisy data at time t.

            if t > 1:
                z = DL_random(x_t.shape, normal=True, seed=seed) #,[0 1] (eqn 3 algorithm 2)
            else:
                z = torch.zeros_like(x_t)

            scalar = 1 / torch.sqrt(self.alphas[t])
            noise_coeff = (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t])
            sigma = torch.sqrt(self.betas[t])
            # sigma = torch.sqrt( ( (1 - self.alpha_bars[t-1]) / (1 - self.alpha_bars[t]) ) * self.betas[t] )

            # print()
            
            x_t_prev = scalar * (x_t.to(self.device) - noise_coeff * model_prediction.to(self.device)) + sigma * z.to(self.device)

            pass
            #####################################
            #          END OF YOUR CODE         #
            #####################################
        
        else:
            ######################################################
            # TODO: Implement a denoising step with thresholding #
            #       Hint: the main difference is how you compute #
            #              the mean of the x_t_prev              #
            #       Hint: 1 call to DL_random                    #
            ######################################################

            if t > 1:
                z = DL_random(x_t.shape, normal=True, seed=seed) #,[0 1] (eqn 3 algorithm 2)
            else:
                z = torch.zeros_like(x_t)
                alpha_bar_prev = 1

            scalar = 1 / torch.sqrt(self.alphas[t])
            noise_coeff = (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t])
            sigma = torch.sqrt(self.betas[t])
            # sigma = torch.sqrt( ( (1 - self.alpha_bars[t-1]) / (1 - self.alpha_bars[t]) ) * self.betas[t] )

            # print()
            
            x_t_prev = scalar * (x_t.to(self.device) - noise_coeff * model_prediction.to(self.device)) + sigma * z.to(self.device)
            
            pass
            ######################################################
            #                  END OF YOUR CODE                  #
            ######################################################

        return x_t_prev
        
    def add_noise(self, original_samples, noise, timesteps):
        """
        add noise to the original samples - the forward diffusion process.  
        Args:
            original_samples (torch.Tensor): the uncorrupted original samples (B, input_shape)
            noise (torch.Tensor): random gaussian noise (B, input_shape)
            timesteps (torch.Tensor): the timesteps for noise addition (B,)
        Returns:
            noisy_samples (torch.Tensor): corrupted samples with amount of noise added based 
                                          on the corresponding timestep (B, input_shape)
        """
        noisy_samples = None
        ###########################################
        # TODO: Implement forward noising process #
        ###########################################

        # print(noise.shape, original_samples.shape)
        # print(timesteps.tolist())
        
        ## Following Algorithm 1 - Training, we need the coeff. that multiplies epsilon_theta

        noisy_samples = torch.ones_like(noise)
        # for t, i in enumerate(timesteps):
        #     # noisy_samples[t] = torch.prod(noise[:t]) + original_samples[:t]
        #     # noisy_samples[t] = self.betas[i] * noise[t] + self.alphas[i] * original_samples[t]
        #     noisy_samples[t] = torch.sqrt(1 - self.alpha_bars[i]) * noise[t] + torch.sqrt(self.alpha_bars[i]) * original_samples[t]
        #     # print(t, i)
        #     # print(noise[:t])
        
        # ## Broadcasting
        # print(self.betas.shape, self.alpha_bars.shape)
        # print(self.alpha_bars)
        # print(self.alpha_bars[timesteps].shape, ';', self.alpha_bars[timesteps].reshape(-1, 1).shape)

        # reshaped_alpha_bars = self.alpha_bars[timesteps].reshape(-1, 1) # ---> doesn't pass GS
        # print(reshaped_alpha_bars.shape)
        s = original_samples.shape
        # print(s, len(s))
        new_size = (-1,) + tuple((len(s) - 1) * [1])
        # print(new_size)
        reshaped_alpha_bars = self.alpha_bars[timesteps].reshape(new_size)
        # print(reshaped_alpha_bars.shape)
        noisy_samples = torch.sqrt(1 - reshaped_alpha_bars) * noise.to(self.device) + torch.sqrt(reshaped_alpha_bars) * original_samples.to(self.device)


        ##########################################
        #          END OF YOUR CODE              #
        ##########################################

        return noisy_samples
    







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

        if DL_random(shape=(1,), normal=False, seed=seed).item() < self.p_uncond:
            cond = torch.zeros_like(cond)  # Set conditioning to zeros for unconditional training

        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################

        loss = None

        ########################################
        # TODO: Implement loss comptutation    #
        # Hint: 2 calls to DL_random           #
        ########################################

        # print('cond:', cond.shape, '; data:', data.shape)
        
        batch_size = cond.shape[0]
        # x0 = DL_random(shape=(batch_size, 1), seed=seed)
        # timesteps = []
        # for i in range(batch_size):
        #     t = DL_random()
        
        t = DL_random(shape=(batch_size,), int_range=[0, self.denoising_steps], normal=False, seed=seed)
        epsilon = DL_random(shape=data.shape, seed=seed)

        noisy_data = self.noise_scheduler.add_noise(data, epsilon, t) # x_t
        # print('noisy_data:', noisy_data.shape)

        # # loss = (torch.norm(noise_GT - noisy_data))^2

        ## From the notebook: use noise_pred_net(sample = x, timestep = t, global_cond = c) to generate an output from the noise prediction network (alternatively you can just call noise_pred_net(x, t, c))
        ep_theta = self.noise_pred_net(noisy_data.to(self.device), t.to(self.device), cond.to(self.device))

        loss = torch.nn.functional.mse_loss(epsilon.to(self.device), ep_theta.to(self.device))


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