from .gradient_optimizer import GradientOptimizer
import logging
import numpy as np
log = logging.getLogger(__name__)

class DeltaBarDelta(GradientOptimizer):
    def __init__(self, theta, mini_k, phi, weights_initialization=None):
        super().__init__()
        self.initialization_method = weights_initialization

        self.optimizer_name = 'Delta Bar Delta'
        log.info(f"Optimizer Name: {self.optimizer_name}")

        self.theta = theta #0.1 #[0,1]
        self.mini_k = mini_k #0.01 # Constant coefficient increment factor
        self.phi = phi #0.1 # [0,1]- Constant learning coefficient decrement factor
        self.dbd_init_process = True

    def optimize(self, **kwargs):
        # --- Weights
        dE_dwij_t = kwargs["dE_dwij_t"] 
        weights = kwargs["weights"]
        mega_delta_weights_array = kwargs["mega_delta_weights_array"]

        # --- Bias
        if self.use_bias:
            dE_dbij_t = kwargs["dE_dbij_t"]
            bias = kwargs["bias"]
            mega_delta_bias_array = kwargs["mega_delta_bias_array"]

        if (self.dbd_init_process == True):
            log.warning("Initialized dbd process")
            self.learning_rates_weights = [np.random.uniform(low=0.05, high=0.2, size=dE_dwij_t[i].shape) for i in range(len(dE_dwij_t))]
            self.learning_rates_weights_deltas = [np.random.uniform(low=0.05, high=0.2, size=dE_dwij_t[i].shape) for i in range(len(dE_dwij_t))]
            self.average_bar_delta_weights = [np.zeros(dE_dwij_t[i].shape) for i in range(len(dE_dwij_t))]
            
            if self.use_bias:
                self.learning_rates_bias = [np.random.uniform(low=0.05, high=0.2, size=dE_dbij_t[i].shape) for i in range(len(dE_dbij_t))]
                self.learning_rates_bias_deltas = [np.random.uniform(low=0.05, high=0.2, size=dE_dbij_t[i].shape) for i in range(len(dE_dbij_t))]
                self.average_bar_delta_bias =[np.zeros(dE_dbij_t[i].shape) for i in range(len(dE_dbij_t))]
                
            self.dbd_init_process = False

        for dw in range(len(dE_dwij_t)): # TODO: Check if this can be expanded to RPROP
            if len(mega_delta_weights_array)>1:
                # delta_bar(t) =  (1-self.theta)*gradient_of_weights + self.theta*previous_gradent of weights  
                self.average_bar_delta_weights[dw]=(1-self.theta)*dE_dwij_t[dw] + self.theta * mega_delta_weights_array[-2][dw]#self.theta*self.average_bar_delta_weights[dw]# self.theta * self.mega_delta_weights_array[-2][dw] #self.theta*self.average_bar_delta_weights[dw] ##
            else:
                self.average_bar_delta_weights[dw]=(1-self.theta)*dE_dwij_t[dw]
            

            # The muliplication of delta(t) * delta(t-1)
            condition_param = self.average_bar_delta_weights[dw] * dE_dwij_t[dw]

            # Looping per layer weight
            # Else condition
            self.learning_rates_weights_deltas[dw] = np.where(condition_param==0, 0.0, self.learning_rates_weights_deltas[dw])
            # Greater than condition
            self.learning_rates_weights_deltas[dw] = np.where(condition_param>0, self.mini_k, self.learning_rates_weights_deltas[dw])
            # Less that condition
            self.learning_rates_weights_deltas[dw] = np.where(condition_param<0, -self.phi*self.learning_rates_weights[dw], self.learning_rates_weights_deltas[dw])

            # Adding the learning deltas to the learning rate arrays
            self.learning_rates_weights[dw] += self.learning_rates_weights_deltas[dw]
            
            # Adjusting the weights
            weights[dw] = weights[dw] - self.learning_rates_weights[dw] *dE_dwij_t[dw]

        if self.use_bias:
            for db in range(len(dE_dbij_t)):
                # Oscillates a bit more when here, when at the end it is smoother just does now work as well
                if len(mega_delta_bias_array)>1: 
                    self.average_bar_delta_bias[db]=(1-self.theta)*dE_dbij_t[db] + self.theta*mega_delta_bias_array[-2][db]#self.theta*self.average_bar_delta_bias[db] #self.theta*self.mega_delta_bias_array[-1][db]
                else:
                    self.average_bar_delta_bias[db]=(1-self.theta)*dE_dbij_t[db]

                # print(self.average_bar_delta_bias)
                bias_condition_param = self.average_bar_delta_bias[db] * dE_dbij_t[db]

                # First condition 
                self.learning_rates_bias_deltas[db] = np.where(bias_condition_param==0, 0.0, self.learning_rates_bias_deltas[db])
                self.learning_rates_bias_deltas[db] = np.where(bias_condition_param>0, self.mini_k, self.learning_rates_bias_deltas[db])
                self.learning_rates_bias_deltas[db] = np.where(bias_condition_param<0, -self.phi*self.learning_rates_bias[db], self.learning_rates_bias_deltas[db])
                
                self.learning_rates_bias[db] += self.learning_rates_bias_deltas[db]
                
                # This is correct
                bias[db] = bias[db] - self.learning_rates_bias[db] * dE_dbij_t[db]
        
        if self.use_bias:
            return weights, bias
        else:
            return weights, 0