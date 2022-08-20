import numpy as np

from nnfs.utils.logs import create_logger
log = create_logger(__name__)

class EarlyStopping:
    """A method that works well to prevent over fitting of the network.

        - E is the error function
        - E_va(t) is teh error on validation set at epoch t
        - E_te(t) is the error on test set(characterizes quality of training)
    """
    def __init__(self, k_epochs=None, pkt_threshold=None, alpha=None):
        if (k_epochs is None) or (pkt_threshold is None) or (alpha is None):
            raise NotImplementedError("k_epochs, pkt_threshold or alpha was not correctly initialized")

        self.k_epochs = k_epochs
        self.pkt_threshold = pkt_threshold
        self.alpha = alpha
        self.early_stopper = "Early stopping activated"

        # With the genetic algorithm we have to take in only the first set of weight

    def training_progress(self, epoch_error_training_plot, pkt_threshold):
        """A moving average calculation that determines how much the average training
        error in a strip n+1 to n+k is larger that the minimum training error in that strip.


        Parameters
        ----------
        epoch_error_training_plot : list
            List containing all of the training losses
        pkt_threshold : float
            Training progress indicator (measure in parts per thousand)

        Returns
        -------
        bool
            True if training should be stopped
        pkt_threshold : float
            Feedback variable to prevent training progress failure for values less than k
        """
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

            # Training progress
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
        """The model will stop training when the generalization loss
        (GL) exceeds a certain threshold (alpha).

        Parameters
        ----------
        epoch_error_validation_plot : list
            List containing all of the validation losses

        Returns
        -------
        bool
            If GL is greater than alpha stop training
        GL
            Saved as part of the model
        """
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
