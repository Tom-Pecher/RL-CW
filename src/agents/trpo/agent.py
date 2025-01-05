# imports.
import torch
from typing import Tuple
import torch.optim as optim

# import from policy_network.py and value_network.py
from .policy_network import PolicyNetwork
from .value_network import ValueNetwork

################################################################################    
# TRPO AGENT (read below its very helpful)
# ------------------------------------------------------------------------------
# if you want to better understand the code, check out the resources below! 
# These resources I used and found very helpful in understanding the algorithms. 
#
# PAPER
# https://people.engr.tamu.edu/guni/csce642/files/trpo.pdf
#
# VIDEOS
# L4 TRPO and PPO (Foundations of Deep RL Series)
# https://www.youtube.com/watch?v=KjWF8VIMGiY
# CS885 Lecture 15a: Trust Region Policy Optimization (Presenter: Shivam Kalra)
# https://www.youtube.com/watch?v=jcF-HaBz0Vw
# Trust Region Policy Optimization | Lecture 78 (Part 2) | Applied Deep Learning
# https://www.youtube.com/watch?v=sCi1Lh_uZ7s
#
# BLOGS
# Trust Region Policy Optimization Explained
# Henry Wu (Mar 9, 2024)
# https://shorturl.at/9K44l
# Generalized Advantage Estimation in Reinforcement Learning (Siwei Causevic - Mar 27, 2023)
# https://shorturl.at/2DjxC
# Spinning Up TRPO (OpenAI)
# https://shorturl.at/VDZwy
#
# STEPS FOLLOWED FOR IMPLEMENTING TRPO : 
# 1. Collect Data 
# 2. Estimate Advantage 
# 3. Surrogate Loss 
# 4. KL Constraint 
# 5. Search Direction 
# 6. Conjugate Gradient 
# 7. Line Search 
# 8. Update Policy Parameters 
# 9. Select Action 
# 10. Update Value Function
################################################################################        
class TRPOAgent:
    def __init__(self, state_dim: int, action_dim: int, max_action: float, device: torch.device):
        """
        initalizes all the hyper parameters and calls both networks 
        """
        # Determines capacity of the network
        UPPER_BOUND = 2
        LOWER_BOUND = -20
        self.policy = PolicyNetwork(state_dim, action_dim, max_action, [64, 64], UPPER_BOUND, LOWER_BOUND).to(device)
        self.value = ValueNetwork(state_dim,[64,64]).to(device)

        # standard parameters for TRPO, 
        # tested the parameters in paper and they work ok too.
        self.GAE_LAMBDA = 0.95
        self.GAMMA = 0.99
        self.MAX_KL = 0.01
        self.MAX_STEP_SIZE = 0.01
        self.BACKTRACK_COEFFICIENT = 0.5
        self.DAMPING = 0.1
        self.VALUE_EPOCHS = 5
        self.ENTROPY_COEFICIENT = 0.01
        self.MAX_BACKTRACKING_DEPTH = 10
        self.LEARNING_RATE = 1e-3
        self.DEVICE_TYPE = device
        self.MAX_ACTION = max_action

        # Adap optimizer 
        learning_rate = self.LEARNING_RATE
        self.value_optimizer = optim.Adam(self.value.parameters(), learning_rate)

    ############################################################################
    # STEP 1: COLLECT DATA 
    ############################################################################
    def collect_data(self, env, min_timesteps):
        """
        get trajectories from environment.
        """
        trajectories = []
        total_step_count = 0

        for _ in range(min_timesteps):
            if total_step_count >= min_timesteps:
                break

            trajectory = {
                'states': [], 
                'actions': [], 
                'rewards':[], 
                'values': [],
                'probabilitys': []
            } 

            state, _ = env.reset()
            done = False

            while done == False:
                # get the state tensor so its actually on the device. 
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.DEVICE_TYPE)
                
                # stop gradient 
                torch.set_grad_enabled(False)

                # get the distribution action and value and probability 
                values = self.value(state_tensor).cpu().item()

                distribution  = self.policy.get_distribution(state_tensor)
                action = distribution.sample()
                log_prob = distribution.log_prob(action).sum( -1)

                # start gradient 
                torch.set_grad_enabled(True)
                probabilitys = log_prob.cpu().item()

                # get the flags
                next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
                # add the trajectory 
                trajectory['states'].append(state)
                trajectory['actions'].append(action.cpu().numpy()[0])
                trajectory['rewards'].append(reward)
                trajectory['values'].append(values)
                trajectory['probabilitys'].append(probabilitys)

                # increment loop 
                total_step_count = total_step_count + 1
                state = next_state

                # end the episode
                if (terminated == True) or (truncated == True): 
                    done = True
                else:
                    done = False


            trajectories.append(trajectory)

        return trajectories

    ############################################################################
    # STEP 2: ESTIMATE ADVANTAGE 
    # 
    # Advantage -
    # the difference in observed reward and the reward that the model would have expected. 
    # A_t = r_t + gamma * GAE advantage at time t * discount factor
    ############################################################################    
    def estimate_advantage(self, trajectory) :
        """
        implements GAE advantage for each step
        """

        # get reward and value for each step in hte trajectory
        rewards = torch.tensor(trajectory['rewards'], dtype=torch.float32).to(self.DEVICE_TYPE)
        values = torch.tensor(trajectory['values'], dtype=torch.float32).to(self.DEVICE_TYPE)

        # setup the advantage and return
        advantages = torch.zeros_like(rewards)
        # returns = torch.zeros_like(rewards)

        last_gae = 0

        # compute the advantage looping from final step to initial step
        t = len(rewards) - 1  # Go backwards through the trajectory 
        while t >= 0:
            if t != len(rewards) - 1:
                # TD error: delta = reward + gamma * V(s') - V(s)
                delta = rewards[t] + self.GAMMA * values[t + 1] - values[t]
                # GAE advantage: last_gae = delta + gamma * lambda * last_gae
                advantages[t] = delta + self.GAMMA * self.GAE_LAMBDA * last_gae
            else:
                # last step so no TD error. 
                # GAE advantage = reward + gamma * lambda * value
                delta = rewards[t] + self.GAMMA * (values[t] * -1)
                advantages[t] = delta + self.GAMMA * self.GAE_LAMBDA * last_gae
            
            last_gae = advantages[t]
            t -= 1 # Iterate backwards

        # noramlise hte advantage. 
        mean = torch.mean(advantages)
        standard_dev = torch.sqrt(torch.mean((advantages - mean) ** 2)) 
        # normalised advantage = (advantage - mean) / (standard_dev + epsilon)
        epsilon = 1e-8
        advantages = (advantages - mean) / (standard_dev + epsilon)

        return advantages, (advantages + values)

    ############################################################################
    # STEP 3: SURROGATE LOSS 
    ############################################################################
    
    def compute_surrogate_loss(self, states, actions, advantages, old_probabilitys):
        """
        optimizes the policy so it can improve but within a bound that is determined 
        by a KL divergence constraint. 

        # L_surrogate - -E_s_a [ratio_old_p_new_p * A_t]
        """

        p_dist_state_s_log_prob = (self.policy.get_distribution(states)).log_prob(actions)

        # ratio_old_p_new_p = exponential(log_p_old(a|_s) - log_p_old(a|_s))
        ratio_old_p_new_p = torch.exp(torch.sum(p_dist_state_s_log_prob, dim=-1) - old_probabilitys)

        # L_surrogate - -E_s_a [ratio_old_p_new_p * A_t]
        return  -1 *(ratio_old_p_new_p * advantages).mean()
    
    ############################################################################
    # STEP 4: KL CONSTRAINT 
    ############################################################################
    def compute_kl_constraint(self, states, p_old):
        """
        stops the agent causing large policy changes 

        # KL divergence = E_s [SUM log(p_new(a|_s)) - log(p_new(a|_s)/p(a|_s_old))]
        """
        # get the new distribution 
        p_new = self.policy.get_distribution(states)

        # under the hood but there is a torch function for it!
        # - kl_div = E_s [SUM log(p_new(a|_s)) - log(p_new(a|_s)/p(a|_s_old))]
        return torch.distributions.kl.kl_divergence(p_old, p_new).mean()
    

    ############################################################################
    # STEP 5: SEARCH DIRECTION 
    ############################################################################

    def fisher_vector_product(self, vector, states, p_old):
        """
        computes the fisher vector product 
        """
        
        # computes the kl constraint using Step 4. and uses it to get the gradient of kl contstraint.
        # torch has a function for this. but under the hood its doing 
        # D_theta(kl)(p_theta_old || p_theta_new) - include in report? 
        grads = torch.autograd.grad(
            self.compute_kl_constraint(states, p_old), self.policy.parameters(),
            create_graph=True,
            retain_graph=True)
        
        # get in one d vector 
        flat_grads_list = []
        for grad in grads:
            flat_grad = grad.view(-1)
            flat_grads_list.append(flat_grad)

        flat_grad_kl = torch.cat(flat_grads_list) + self.DAMPING * vector

        # approx fisher matrix = g * v 
        g_v = (flat_grad_kl * vector)
        grad_vector = (g_v).sum()

        grad_grads = torch.autograd.grad(grad_vector, self.policy.parameters(), retain_graph=True)

        # get in one d vector 
        fisher_product_list = []
        for grad in grad_grads:
            fisher_product_list.append(grad.contiguous().view(-1))

        # fisher product =  D_theta(kl)(p_theta_old || p_theta_new) * v + damping
        return torch.cat(fisher_product_list) + self.DAMPING * vector

    def compute_search_direction(self, states: torch.Tensor, actions: torch.Tensor, 
                            advantages: torch.Tensor, old_probabilitys: torch.Tensor):
        """
        computes natural gradient using 
        conjugate gradient and makes use of the fisher equation for computing the gradient
        """
        p_old = self.policy.get_distribution(states)

        # compute the surrogate loss using Step 3. and use it to the surrogate loss gradient 
        grads = torch.autograd.grad(self.compute_surrogate_loss(states, actions, advantages, old_probabilitys),
                                    self.policy.parameters())
        
        # convert back into a 1d vector again 
        g = [grad.view(-1) for grad in grads]
        F_x = torch.cat(g) * -1 

        # Solve the Ax = -b equation using conjugate gradient
        step_direction = self.conjugate_gradient(lambda v: self.fisher_vector_product(v, states, p_old), F_x)
        return step_direction
    
    ############################################################################
    # STEP 6: CONJUGATE GRADIENT 
    #
    # I kept getting logic errors in my implementation of this so i used the below resource to help
    # debug it, explained in README.
    # https://shorturl.at/a5PZZ
    # Vladyslav Yazykov (May 24, 2020), Accessed Date (December 2024)
    ############################################################################
    def conjugate_gradient(self, k_matrix_vector_product, b, max_iterations=10, residual_tol=1e-6):
        """
        calculates the conjugate gradient for each step in the search direction
        """
        # Ax = b
        # Gaol - f(x) = 1/2 * x^T * A * x - b^T * x

        # initialise x, r, p
        
        # initial guess Ax0 = 0
        Ax0 = torch.zeros_like(b)

        # r0 = b - Ax0 = b 
        # residual_0 = copy.deepcopy(b)
        residual_0 = b.clone()

        # p0 = r0
        # p_k_next = copy.deepcopy(residual_0)
        p_k_next = residual_0.clone()
        epsilon = 1e-6

        # iterate over the max iterations   
        for _ in range(max_iterations):
            # Apk = A * pk
            Apk = k_matrix_vector_product(p_k_next)

            # alpha = (T_kr_k) 
            #        ----------
            #     (T_pk_Apk) + epsilon
            rT_kr_k = torch.matmul(residual_0, residual_0)
            pT_Apk = torch.matmul(p_k_next, Apk) +epsilon
            alpha = rT_kr_k / (pT_Apk)

            # Ax0 = Ax0 + alpha * p
            Ax0 = Ax0 + alpha * p_k_next

            # update residual for next iteration 
            # r_new = r_0 - alpha * Apk
            residual_k_new = residual_0 - alpha * Apk

            # if r_new.norm() < residual_tol:
            if residual_k_new.norm() >residual_tol:
                # beta = rT_k+1 * r_k+1
                #          ---------
                #      rT_k * r_k + epsilon
                r_k_new_r_k_new = torch.matmul(residual_k_new, residual_k_new)
                r_0_r_0 = torch.matmul(residual_0, residual_0)
                beta_k = r_k_new_r_k_new/ (r_0_r_0 + epsilon)

                # p_k+1 = r_k+1 + beta * p_k
                p_k_next = residual_k_new + beta_k *p_k_next
                # r_0 = r_k+1
                residual_0 = residual_k_new
            else:
                break

        return Ax0

    ############################################################################
    # STEP 7: LINE SEARCH 
    ############################################################################
    def line_search(self, states, actions, advantages, old_probabilitys, step_dir):
        """
        find optimal step size for the optimal polcy update
        this will find the largest step that can be taken without violating the KL constraint.
        """
        step_size = self.MAX_STEP_SIZE
        max_depth = self.MAX_BACKTRACKING_DEPTH

        # get the surrogate loss done in Step 3. 
        old_loss = self.compute_surrogate_loss(states, actions, advantages, old_probabilitys)

        # get the old params and distribution
        old_parameters = torch.cat([p.data.view(-1) for p in self.policy.parameters()])
        old_dist = self.policy.get_distribution(states)

        for _ in range(max_depth):
            # simply theta_new = old + step_size * step_dir
            new = old_parameters + step_size * step_dir
            # update parameters STEP 8. 
            self.update_policy_params(new)

            # compute the surrogate loss and kl constraint using Step 3. and Step 4. 
            new_loss = self.compute_surrogate_loss(states, actions, advantages, old_probabilitys)
            kl = self.compute_kl_constraint(states, old_dist)

            # only accept if the kl is smaller than the max kl and loss is not smaller than 0 otherwise continute to backtrack. 
            if kl <= self.MAX_KL and (old_loss - new_loss) > 0: return step_size
            step_size *= self.BACKTRACK_COEFFICIENT
            
            return step_size
        return 0

    ############################################################################
    # STEP 8: UPDATE POLICY PARAMETERS 
    ############################################################################
    def update_policy_params(self, new_params):
        """
        updates the policy parameters 
        """
        index = 0
        # this is quite slow but it works. 
        # iterates over old parameters and sets old ones as new ones 
        for each_param in self.policy.parameters():
            # gets the number of elements in a tensor parameter
            tot_size = each_param.numel()

            # slice, reshape and copy the new parameter into the old one 
            sliced_new_params = new_params[index:index + tot_size]
            reshaped_params = sliced_new_params.view(each_param.size())
            each_param.data.copy_(reshaped_params)

            index = index + tot_size

    ############################################################################
    # STEP 9: SELECT ACTION 
    ############################################################################    
    def select_action(self, state, evaluate=False):
        """
        determines the action from the policy, similar to the init intial action
        """
        with torch.no_grad(): # torch.set_grad_enabled(False)
            # state into a tensor 
            state = torch.FloatTensor(state).unsqueeze(0).to(self.DEVICE_TYPE)
            dist = self.policy.get_distribution(state)

            if evaluate:
                action = dist.mean()
            else:
                action = dist.sample()    
            # return action as array
            return action.cpu().numpy()[0]
        

    ############################################################################
    # STEP 10: UPDATE VALUE FUNCTION 
    ############################################################################
    def update_value(self, states, target):
        """
        gets the the min of the loss, backpropogates it 
        normalises the gradient to update the value.  

        EquationL Loss: 
        Value loss = 1/N * Sum(state_value_pred - state_value_target)^2
        """
        # convert to tensors and reshape to col vector 
        states = torch.FloatTensor(states).to(self.DEVICE_TYPE)
        target = torch.FloatTensor(target).unsqueeze(1).to(self.DEVICE_TYPE)

        # iterate over the value epochs 
        value_epoch_count = 0
        while value_epoch_count < self.VALUE_EPOCHS:
            prediction = self.value(states)

            # Value loss = 1/N * Sum(state_value_pred - state_value_target)^2
            # value loss mean of squared diff between pred and target 
            value_loss_squared = (prediction - target).pow(2)
            self.value_optimizer.zero_grad()
            ((value_loss_squared).mean()).backward() # backpropagate the loss 

            # normalise, gradinet         
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), 5.0)
            self.value_optimizer.step()
            
            value_epoch_count = value_epoch_count + 1
