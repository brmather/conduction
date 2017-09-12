
import numpy as np

class InvObservation(object):
    """
    Inversion Observation for use in an objective function

    This requires an interpolation object with information on 
    """

    def __init__(self, interp, obs, obs_err, obs_coords=None):
        """
        Arguments
        ---------
         obs        : observations
         obs_err    : observational uncertainty
         obs_coords : coordinates in Cartesian N-dimensional space
                    : (optional) ndarray shape (n, dim)
        """
        self.v = obs
        self.dv = obs_err
        self.coords = obs_coords

        self.interp = interp

        self.gweight = self.ghost_weights


    def ghost_weights(self):

        interp = self.interp

        w = interp(self.coords)
        w = 1.0/np.floor(w + 1e-12)
        offproc = np.isnan(w)
        w[offproc] = 0.0 # these are weighted with zeros

        return w


class InvPrior(object):

    def __init__(self, prior, prior_err):

        self.v = prior
        self.dv = prior_err
        self.gweight = 1.0

