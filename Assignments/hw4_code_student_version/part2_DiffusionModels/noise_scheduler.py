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

            print()
            
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
        noisy_samples = torch.sqrt(1 - reshaped_alpha_bars) * noise + torch.sqrt(reshaped_alpha_bars) * original_samples


        ##########################################
        #          END OF YOUR CODE              #
        ##########################################

        return noisy_samples