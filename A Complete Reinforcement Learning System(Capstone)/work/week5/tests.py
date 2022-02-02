import numpy as np

class TrueAdam():
    def __init__(self, layer_sizes, 
                 optimizer_info):
        self.layer_sizes = layer_sizes

        # Specify Adam algorithm's hyper parameters
        self.step_size = optimizer_info.get("step_size")
        self.beta_m = optimizer_info.get("beta_m")
        self.beta_v = optimizer_info.get("beta_v")
        self.epsilon = optimizer_info.get("epsilon")
        
        # Initialize Adam algorithm's m and v
        self.m = [dict() for i in range(1, len(self.layer_sizes))]
        self.v = [dict() for i in range(1, len(self.layer_sizes))]
        
        for i in range(0, len(self.layer_sizes) - 1):
            self.m[i]["W"] = np.zeros((self.layer_sizes[i], self.layer_sizes[i+1]))
            self.m[i]["b"] = np.zeros((1, self.layer_sizes[i+1]))
            self.v[i]["W"] = np.zeros((self.layer_sizes[i], self.layer_sizes[i+1]))
            self.v[i]["b"] = np.zeros((1, self.layer_sizes[i+1]))
        
        # Initialize beta_m_product and beta_v_product to be later used for computing m_hat and v_hat
        self.beta_m_product = self.beta_m
        self.beta_v_product = self.beta_v
    
    def update_weights(self, weights, td_errors_times_gradients):
        """
        Given weights and update g, return updated weights
        """
        
        for i in range(len(weights)):
            for param in weights[i].keys():
                ### update self.m and self.v
                self.m[i][param] = self.beta_m * self.m[i][param] + (1 - self.beta_m) * \
                    td_errors_times_gradients[i][param]
                self.v[i][param] = self.beta_v * self.v[i][param] + (1 - self.beta_v) * \
                    (td_errors_times_gradients[i][param] ** 2)

                ### compute m_hat and v_hat
                m_hat = self.m[i][param] / (1 - self.beta_m_product)
                v_hat = self.v[i][param] / (1 - self.beta_v_product)

                ### update weights
                weight_update = self.step_size * m_hat / (np.sqrt(v_hat) + self.epsilon)
                weights[i][param] = weights[i][param] + weight_update
                
        ### update self.beta_m_product and self.beta_v_product
        self.beta_m_product *= self.beta_m
        self.beta_v_product *= self.beta_v
        
        return weights
    
def __true__softmax(action_values, tau=0.001):
    preferences = action_values / tau
    max_preference = np.max(preferences, axis = 1).reshape((-1, 1))
    exp_preferences = np.exp(preferences - max_preference)
    action_probs = exp_preferences / np.sum(exp_preferences, axis = 1).reshape((-1, 1))
    action_probs = action_probs.squeeze()
    return action_probs