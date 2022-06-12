import numpy as np
import logging

from utils.logs import create_logger
log = create_logger(__name__)

class EarlyStopping:
    def __init__(self, k_epochs=None, pkt_threshold=None, alpha=None):
        if (k_epochs is None) or (pkt_threshold is None) or (alpha is None):
            raise NotImplementedError("k_epochs, pkt_threshold or alpha was not correctly initialized")

        self.k_epochs = k_epochs
        self.pkt_threshold = pkt_threshold
        self.alpha = alpha
        self.early_stopper = "Early stopping activated"

        # With the genetic algorithm we have to take in only the first weight

    def training_progress(self, epoch_error_training_plot, pkt_threshold):
        # TODO: Make sure that the optimal weight is saved at the minimum
        k_epochs = self.k_epochs
        try:
            numerator = 0
            denominator = 0
            denom_check_arr= []
            for i in range(1,k_epochs+1):
                store = epoch_error_training_plot[-i]
                numerator += store
                denom_check_arr.append(store)

            denominator = k_epochs*np.min(denom_check_arr)

            P_k_t = 1000*((numerator/denominator)-1)
            log.info(f"P_k_t: {P_k_t}")

            if (P_k_t < pkt_threshold):
                return True, pkt_threshold
            else:
                return False, pkt_threshold

        except:
            log.warning("Training has not been for long enough")
            return False, pkt_threshold

    def early_stopping(self, epoch_error_validation_plot):
        # page 24
        # Call this function in the loop you want it to stop

        E_opt = np.min(epoch_error_validation_plot)

        # Generalization Loss
        GL = 100 * ((epoch_error_validation_plot[-1]/E_opt) - 1)
        log.info(f"GL: {GL}")
        if (self.alpha < GL):
            return True, GL
        else:
            return False, GL
