import numpy as np
from collections import deque

class SPSA:

    def __init__(self, seed, lr, fb_type, alpha=0.5, gamma=0.1, N_iters=1e3):
        np.random.seed(seed)

        # SPSA variables
        self.alpha = alpha
        self.gamma = gamma
        self.A = 0.1*N_iters
        self.c = 1
        self.epsilon = np.random.randn()
        magnitude_g0 = 1
        self.a = ((self.A+1)**alpha)/magnitude_g0

        # beta variables
        self.k = 1
        self.lr = lr
        self.optimism = np.random.uniform(low=-1, high=1)
        
        # feedback variables
        self.error_buffer = deque(maxlen=5)
        self.use_std = True
        self.fb_type = fb_type
        self.decay = 1 - lr

    def sample(self,):
        
        self.epsilon = np.random.randn()
        return self.optimism + self.epsilon

    def update(self, curr_episode_return, prev_episode_return):

        if 'divfb' in self.fb_type:
            if prev_episode_return == 0:
                feedback = 1
            else:
                feedback = curr_episode_return/prev_episode_return
                feedback -= 1
        elif 'subfb' in self.fb_type:
            feedback = curr_episode_return - prev_episode_return

        self.error_buffer.append(feedback)
        
        # normalize
        feedback -= np.mean(self.error_buffer)
        if self.use_std and len(self.error_buffer) > 1:
            norm = np.std(self.error_buffer)
            feedback /= norm

        """
        # Divisive fb            
        gk = self.epsilon*np.tanh(return_fb)
        """
        
        # Orig fb
        gk = self.epsilon*(feedback)

        if 'wlrd' in self.fb_type:
            #ak = self.a/((self.k+self.A)**self.alpha)
            ck = self.c/(self.k**self.gamma)
            self.optimism += self.lr*ck*gk
        else:
            if self.k > 1:
                self.optimism *= self.decay
            self.optimism += self.lr*gk
        self.k += 1