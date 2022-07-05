# diffusion categorical pytorch version of D3PM  https://arxiv.org/pdf/2107.03006.pdf
import json
from random import random
import re
from pyparsing import alphas
import random
import rdkit
from rdkit import Chem
import scipy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

disc_data = []

def meanflat(x):
    """Take the mean over all axes except the first batch dimension."""
    return x.mean(axis=tuple(range(1, len(x.shape))))


def log_min_exp(a, b, epsilon=1.e-6):
    """Computes the log(exp(a) - exp(b)) (b<a) in a numerically stable fashion."""
    y = a + torch.log1p(-torch.exp(b - a) + epsilon)
    return y


def categorical_kl_probs(probs1, probs2, eps=1.e-6):
    """KL divergence between categorical distributions.

  Distributions parameterized by logits.

  Args:
    probs1: probs of the first distribution. Last dim is class dim.
    probs2: probs of the second distribution. Last dim is class dim.
    eps: float small number to avoid numerical issues.

  Returns:
    KL(C(probs) || C(logits2)): shape: logits1.shape[:-1]
  """
    out = probs1 * (torch.log(probs1 + eps) - torch.log(probs2 + eps))
    return torch.sum(out, dim=-1)


def categorical_kl_logits(logits1, logits2, eps=1.e-6):
    """KL divergence between categorical distributions.

  Distributions parameterized by logits.

  Args:
    logits1: logits of the first distribution. Last dim is class dim.
    logits2: logits of the second distribution. Last dim is class dim.
    eps: float small number to avoid numerical issues.

  Returns:
    KL(C(logits1) || C(logits2)): shape: logits1.shape[:-1]
  """
    out = (
            F.softmax(logits1 + eps, dim=-1) *
            (F.log_softmax(logits1 + eps, dim=-1) -
             F.log_softmax(logits2 + eps, dim=-1)))
    return torch.sum(out, dim=-1)


def categorical_log_likelihood(x, logits):
    """Log likelihood of a discretized Gaussian specialized for image data.

  Assumes data `x` consists of integers [0, num_classes-1].

  Args:
    x: where to evaluate the distribution. shape = (bs, ...), dtype=int32/int64
    logits: logits, shape = (bs, ..., num_classes)

  Returns:
    log likelihoods
  """
    log_probs = F.log_softmax(logits, dim=-1)
    x_onehot = F.one_hot(x, logits.shape[-1])
    return torch.sum(log_probs * x_onehot, dim=-1)


def get_diffusion_betas(type, num_timesteps, linear_type="uniform", s=0.008):
    """Get betas from the hyperparameters."""
    if type == 'linear':
        # Used by Ho et al. for DDPM, https://arxiv.org/abs/2006.11239.
        # To be used with Gaussian diffusion models in continuous and discrete
        # state spaces.
        # To be used with transition_mat_type = 'gaussian'
        # start: 1e-4 gauss, 0.02 uniform
        # stop: 0.02, gauss, 1. uniform
        if linear_type == "uniform":
            start, stop = 0.02, 1.
        elif linear_type == "gauss":
            start, stop = 1e-4, 0.02
        else:
            raise NotImplementedError(linear_type)
        return np.linspace(start, stop, num_timesteps)
    elif type == 'cosine':
        # Schedule proposed by Hoogeboom et al. https://arxiv.org/abs/2102.05379
        # To be used with transition_mat_type = 'uniform'.
        steps = (
                np.arange(num_timesteps + 1, dtype=np.float64) /
                num_timesteps)
        alpha_bar = np.cos((steps + 0.008) / 1.008 * np.pi / 2)
        betas = np.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], 0.999)
        return betas, alpha_bar
    elif type == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        # Proposed by Sohl-Dickstein et al., https://arxiv.org/abs/1503.03585
        # To be used with absorbing state models.
        # ensures that the probability of decaying to the absorbing state
        # increases linearly over time, and is 1 for t = T-1 (the final time).
        # To be used with transition_mat_type = 'absorbing'
        return 1. / np.linspace(num_timesteps, 1., num_timesteps)
    else:
        raise NotImplementedError(type)


class CategoricalDiffusion(nn.Module):
    def __init__(self, num_classes, shape, denoise_fn, timesteps=1000,
                 model_output="logits", beta_schedule="cosine",
                 loss_type='hybrid', transition_mat_type="uniform", transition_bands=None, hybrid_coeff=0.,
                 model_prediction='x_start', device="cuda"):
        super(CategoricalDiffusion, self).__init__()
        assert loss_type in ('hybrid', 'cross_entropy_x_start', 'kl')
        assert model_output in ("logistic_pars", "logits")
        assert model_prediction in ('x_start', 'xprev')

        self.num_classes = num_classes
        self.shape = shape
        self.device = device
        self._denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.model_output = model_output
        self.num_timesteps = timesteps
        self.model_prediction = model_prediction
        self.hybrid_coeff = hybrid_coeff
        self.transition_mat_type = transition_mat_type
        self.transition_bands = transition_bands    # 类只能在邻域类别内转变

        self.eps = 1.e-6
        betas = get_diffusion_betas(beta_schedule, self.num_timesteps)
        if beta_schedule == "cosine":
            betas, self.alpha_bar = betas
        self.betas = betas.astype(np.float64)
        self.alpha_bar = torch.from_numpy(self.alpha_bar).to(device)

        if self.transition_mat_type == 'uniform':
            q_one_step_mats = [self._get_transition_mat(t)
                               for t in range(0, self.num_timesteps)]
                               
        elif self.transition_mat_type == 'gaussian':
            q_one_step_mats = [self._get_gaussian_transition_mat(t)
                               for t in range(0, self.num_timesteps)]
        elif self.transition_mat_type == 'absorbing':
            q_one_step_mats = [self._get_absorbing_transition_mat(t)
                               for t in range(0, self.num_timesteps)]
        else:
            raise ValueError(
                f"transition_mat_type must be 'gaussian', 'uniform', 'absorbing' "
                f", but is {self.transition_mat_type}"
            )

        self.betas = torch.from_numpy(self.betas)
        self.q_onestep_mats = np.stack(q_one_step_mats, axis=0)
        # self.cond_q_onestep_mats = 
        assert self.q_onestep_mats.shape == (self.num_timesteps,
                                             self.num_classes,
                                             self.num_classes)
        

        # Construct transition matrices for q(x_t|x_start)
        q_mat_t = self.q_onestep_mats[0]
        q_mats = [q_mat_t]
        for t in range(1, self.num_timesteps):
            # Q_{1...t} = Q_{1 ... t-1} Q_t = Q_1 Q_2 ... Q_t
            q_mat_t = np.tensordot(q_mat_t, self.q_onestep_mats[t],
                                   axes=[[1], [0]])
            q_mats.append(q_mat_t)
        self.q_mats = np.stack(q_mats, axis=0)
        assert self.q_mats.shape == (self.num_timesteps, self.num_classes,
                                     self.num_classes), self.q_mats.shape
        self.q_mats = torch.from_numpy(self.q_mats).to(self.device)
        # Don't precompute transition matrices for q(x_{t-1} | x_t, x_start)
        # Can be computed from self.q_mats and self.q_one_step_mats.
        # Only need transpose of q_onestep_mats for posterior computation.
        self.transpose_q_onestep_mats = torch.from_numpy(np.transpose(self.q_onestep_mats,
                                                                      axes=(0, 2, 1))).to(self.device)
        del self.q_onestep_mats

    def _get_transition_mat(self, t):
        r"""Computes transition matrix for q(x_t|x_{t-1}).

            This method constructs a transition
            matrix Q with
            Q_{ij} = beta_t / self.num_classes       if |i-j| <= self.transition_bands
                     1 - \sum_{l \neq i} Q_{il} if i==j.
                     0                          else.

            Args:
              t: timestep. integer scalar (or numpy array?)

            Returns:
              Q_t: transition matrix. shape = (self.num_classes, self.num_classes).
            """
        if self.transition_bands is None:
            return self._get_full_transition_mat(t)
        # Assumes num_off_diags < self.num_classes
        beta_t = self.betas[t]

        mat = np.zeros((self.num_classes, self.num_classes),
                       dtype=np.float64)
        off_diag = np.full(shape=(self.num_classes - 1,),
                           fill_value=beta_t / float(self.num_classes),
                           dtype=np.float64)
        for k in range(1, self.transition_bands + 1):
            mat += np.diag(off_diag, k=k)
            mat += np.diag(off_diag, k=-k)
            off_diag = off_diag[:-1]

        # Add diagonal values such that rows sum to one.
        diag = 1. - mat.sum(1)
        mat += np.diag(diag, k=0)
        return mat

    def _get_domain_transition_mat(self, t):
        if self.transition_bands is None:
            return self._get_full_transition_mat(t)
        # Assumes num_off_diags < self.num_classes
        beta_t = self.betas[t]

        mat = np.zeros((self.num_classes, self.num_classes),
                       dtype=np.float64)
        off_diag = np.full(shape=(self.num_classes - 1,),
                           fill_value=beta_t / float(self.num_classes),
                           dtype=np.float64)
        for k in range(1, self.transition_bands + 1):
            mat += np.diag(off_diag, k=k)
            mat += np.diag(off_diag, k=-k)
            off_diag = off_diag[:-1]

        diag = 1. - mat.sum(1)
        mat += np.diag(diag, k=0)
        return mat

    def _get_gaussian_transition_mat(self, t):
        r"""Computes transition matrix for q(x_t|x_{t-1}).

            This method constructs a transition matrix Q with
            decaying entries as a function of how far off diagonal the entry is.
            Normalization option 1:
            Q_{ij} =  ~ softmax(-val^2/beta_t)   if |i-j| <= self.transition_bands
                     1 - \sum_{l \neq i} Q_{il}  if i==j.
                     0                          else.

            Normalization option 2:
            tilde{Q}_{ij} =  softmax(-val^2/beta_t)   if |i-j| <= self.transition_bands
                             0                        else.

            Q_{ij} =  tilde{Q}_{ij} / sum_l{tilde{Q}_{lj}}

            Args:
              t: timestep. integer scalar (or numpy array?)

            Returns:
              Q_t: transition matrix. shape = (self.num_classes, self.num_classes).
            """
        transition_bands = self.transition_bands if self.transition_bands else self.num_classes - 1

        beta_t = self.betas[t]

        mat = np.zeros((self.num_classes, self.num_classes),
                       dtype=np.float64)

        # Make the values correspond to a similar type of gaussian as in the
        # gaussian diffusion case for continuous state spaces.
        values = np.linspace(start=0., stop=255., num=self.num_classes,
                             endpoint=True, dtype=np.float64)
        values = values * 2. / (self.num_classes - 1.)
        values = values[:transition_bands + 1]
        values = -values * values / beta_t

        values = np.concatenate([values[:0:-1], values], axis=0)
        values = scipy.special.softmax(values, axis=0)
        values = values[transition_bands:]
        for k in range(1, transition_bands + 1):
            off_diag = np.full(shape=(self.num_classes - k,),
                               fill_value=values[k],
                               dtype=np.float64)

            mat += np.diag(off_diag, k=k)
            mat += np.diag(off_diag, k=-k)

        # Add diagonal values such that rows and columns sum to one.
        # Technically only the ROWS need to sum to one
        # NOTE: this normalization leads to a doubly stochastic matrix,
        # which is necessary if we want to have a uniform stationary distribution.
        diag = 1. - mat.sum(1)
        mat += np.diag(diag, k=0)

        return mat

    def _get_absorbing_transition_mat(self, t):  # 图片该列元素所有时间t都被mask
        """Computes transition matrix for q(x_t|x_{t-1}).

            Has an absorbing state for pixelvalues self.num_classes//2.

            Args:
              t: timestep. integer scalar.

            Returns:
              Q_t: transition matrix. shape = (self.num_classes, self.num_classes).
            """
        beta_t = self.betas[t]

        diag = np.full(shape=(self.num_classes,), fill_value=1. - beta_t,
                       dtype=np.float64)
        mat = np.diag(diag, k=0)
        # Add beta_t to the self.num_classes/2-th column for the absorbing state.
        # raise NotImplementedError
        mat[:, self.num_classes // 2] += beta_t
        # mat[:, 0] += beta_t     # padding

        return mat

    def _get_full_transition_mat(self, t):
        """Computes transition matrix for q(x_t|x_{t-1}).

            Contrary to the band diagonal version, this method constructs a transition
            matrix with uniform probability to all other states.

            Args:
              t: timestep. integer scalar.

            Returns:
              Q_t: transition matrix. shape = (self.num_classes, self.num_classes).
            """
        beta_t = self.betas[t]
        mat = np.full(shape=(self.num_classes, self.num_classes),
                      fill_value=beta_t / float(self.num_classes),
                      dtype=np.float64)
        diag_indices = np.diag_indices_from(mat)
        diag_val = 1. - beta_t * (self.num_classes - 1.) / self.num_classes
        mat[diag_indices] = diag_val
        return mat

    def _at(self, a, t, x):
        """Extract coefficients at specified timesteps t and conditioning data x.

    Args:
      a: np.ndarray: plain NumPy float64 array of constants indexed by time.
      t: torch.ndarray: Jax array of time indices, shape = (batch_size,).
      x: torch.ndarray: jax array of shape (bs, ...) of int32 or int64 type.
        (Noisy) data. Should not be of one hot representation, but have integer
        values representing the class values.

    Returns:
      a[t, x]: torch.ndarray: Jax array.
    """
        a = torch.as_tensor(a, dtype=torch.float32)
        t_broadcast = t.reshape((len(t), *(1 for i in range(1, len(x.shape)))))

        # x.shape = (bs, s)
        # t_broadcast_shape = (bs, 1)
        # a.shape = (num_timesteps, self.num_classes, self.num_classes)
        # out.shape = (bs, s, self.num_classes)
        # out[i, j, m] = a[t[i, j], x[i, j], m]
        return a[t_broadcast, x]

    def _at_onehot(self, a, t, x):
        """Extract coefficients at specified timesteps t and conditioning data x.

    Args:
      a: np.ndarray: plain NumPy float64 array of constants indexed by time.
      t: torch.ndarray: Jax array of time indices, shape = (bs,).
      x: torch.ndarray: jax array, shape (bs, ..., self.num_classes), float32 type.
        (Noisy) data. Should be of one-hot-type representation.

    Returns:
      out: torch.ndarray: Jax array. output of dot(x, a[t], axis=[[-1], [1]]).
        shape = (bs, ..., self.num_classes)
    """
        a = torch.as_tensor(a, dtype=torch.float32)

        # x.shape = (bs,s, self.num_classes)
        # a[t]shape = (bs, self.num_classes, self.num_classes)
        # out.shape = (bs, s, self.num_classes)
        xx = torch.matmul(x, a[t, Ellipsis])
        return xx

    def q_probs(self, x_start, t):
        """Compute probabilities of q(x_t | x_start).

    Args:
      x_start: torch.ndarray: jax array of shape (bs, ...) of int32 or int64 type.
         Should not be of one hot representation, but have integer values
         representing the class values.
      t: torch.ndarray: jax array of shape (bs,).

    Returns:
      probs: torch.ndarray: jax array, shape (bs, x_start.shape[1:],
                                            self.num_classes).
    """
        return self._at(self.q_mats, t, x_start)

    def q_sample(self, x_start,  t, noise):
        """Sample from q(x_t | x_start) (i.e. add noise to the data).

    Args:
      x_start: torch.array: original clean data, in integer form (not onehot).
        shape = (bs, ...).
      t: :torch.array: timestep of the diffusion process, shape (bs,).
      noise: torch.ndarray: uniform noise on [0, 1) used to sample noisy data.
        Should be of shape (*x_start.shape, self.num_classes).

    Returns:
      sample: torch.ndarray: same shape as x_start. noisy data.
    """
        assert noise.shape == x_start.shape + (self.num_classes,), f"{noise.shape}, {x_start.shape + (self.num_classes,)}"
        logits = torch.log(self.q_probs(x_start, t) + self.eps)

        # To avoid numerical issues clip the noise to a minimum value
        noise = torch.clip(noise, min=torch.finfo(noise.dtype).tiny, max=1.).to(self.device)
        gumbel_noise = - torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, dim=-1)

    def q_sample_for_classifier(self, x_start,  t, noise):
        assert noise.shape == x_start.shape + (self.num_classes,), f"{noise.shape}, {x_start.shape + (self.num_classes,)}"
        logits = torch.log(self.q_probs(x_start, t) + self.eps)

        # To avoid numerical issues clip the noise to a minimum value
        # noise = torch.clip(noise, min=torch.finfo(noise.dtype).tiny, max=1.).to(self.device)
        # gumbel_noise = - torch.log(-torch.log(noise))
        return logits

    def _get_logits_from_logistic_pars(self, loc, log_scale):
        """Computes logits for an underlying logistic distribution."""

        loc = loc.reshape(*(loc.shape, 1))
        log_scale = log_scale.reshape(*(log_scale.shape, 1))

        # Shift log_scale such that if it's zero the probs have a scale
        # that is not too wide and not too narrow either.
        inv_scale = torch.exp(- (log_scale - torch.full(log_scale.shape, 2.)))

        bin_width = 2. / (self.num_classes - 1.)
        bin_centers = torch.linspace(start=-1., end=1., steps=self.num_classes, dtype=torch.float32)

        bin_centers = bin_centers.reshape(*(1 for i in range(0, len(loc.shape) - 1)), *bin_centers.shape)

        bin_centers = bin_centers - loc
        log_cdf_min = nn.LogSigmoid()(
            inv_scale * (bin_centers - 0.5 * bin_width))
        log_cdf_plus = nn.LogSigmoid()(
            inv_scale * (bin_centers + 0.5 * bin_width))

        logits = log_min_exp(log_cdf_plus, log_cdf_min, self.eps)

        # Normalization:
        # # Option 1:
        # # Assign cdf over range (-\inf, x + 0.5] to pmf for pixel with
        # # value x = 0.
        # logits = logits.at[..., 0].set(log_cdf_plus[..., 0])
        # # Assign cdf over range (x - 0.5, \inf) to pmf for pixel with
        # # value x = 255.
        # log_one_minus_cdf_min = - F.softplus(
        #     inv_scale * (bin_centers - 0.5 * bin_width))
        # logits = logits.at[..., -1].set(log_one_minus_cdf_min[..., -1])
        # # Option 2:
        # # Alternatively normalize by reweighting all terms. This avoids
        # # sharp peaks at 0 and 255.
        # since we are outputting logits here, we don't need to do anything.
        # they will be normalized by softmax anyway.

        return logits

    def q_posterior_logits(self, x_start, x_t, t, x_start_logits):
        """
        math::

        """
        """Compute logits of q(x_{t-1} | x_t, x_start)."""

        if x_start_logits:  # 这是从模型输出的结果，是one-hot的float类型
            assert x_start.shape == x_t.shape + (self.num_classes,), (
                x_start.shape, x_t.shape)
        else:
            assert x_start.shape == x_t.shape, (x_start.shape, x_t.shape)

        fact1 = self._at(self.transpose_q_onestep_mats, t, x_t)  # x_t QTt. x_t Long type
        if x_start_logits:
            fact2 = self._at_onehot(self.q_mats, t - 1, 
                                    F.softmax(x_start, dim=-1)) # 
            tzero_logits = x_start
        else:
            fact2 = self._at(self.q_mats, t - 1, x_start)  # x_0 \bar{Q}t-1
            tzero_logits = torch.log(
                F.one_hot(x_start, num_classes=self.num_classes)
                + self.eps)

        # At t=0 we need the logits of q(x_{-1}|x_0, x_start)
        # where x_{-1} == x_start. This should be equal the log of x_0.
        tzero_logits = tzero_logits.to(device=x_start.device)
        # qprob = self.q_probs(x_start, t).to(device=x_start.device)
        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        out = out.to(device=x_start.device)
        t_broadcast = t.reshape((len(t), *(1 for i in range(1, len(out.shape))))).to(device=x_start.device)
        return torch.where(t_broadcast == 0, tzero_logits,
                           out)

    def p_logits(self, model_fn,  *, x, t, cond=None):
        """Compute logits of p(x_{t-1} | x_t)."""
        assert t.shape == (x.shape[0],)

        model_output = model_fn(x, t, cond=cond)
        if model_output.size(1) == 84:
            model_output = model_output[:, :-4, :]


        if self.model_output == 'logits':
            model_logits = model_output

        elif self.model_output == 'logistic_pars':
            # Get logits out of discretized logistic distribution.
            loc, log_scale = model_output
            model_logits = self._get_logits_from_logistic_pars(loc, log_scale)

        else:
            raise NotImplementedError(self.model_output)

        if self.model_prediction == 'x_start':
            # Predict the logits of p(x_{t-1}|x_t) by parameterizing this distribution
            # as ~ sum_{pred_x_start} q(x_{t-1}, x_t |pred_x_start)p(pred_x_start|x_t)
            pred_x_start_logits = model_logits

            x_tm1 = self.q_posterior_logits(pred_x_start_logits, x, t, x_start_logits=True)

            # if kwargs.get("cond_fn") and int (t[0] > kwargs.get("cond_step")):    # 这取消注释
            #     x_tm1 = self.condition_mean(x_tm1, t-1, **kwargs)

            t_broadcast = t.reshape((len(t), *(1 for i in range(1, len(model_logits.shape))))).to(device=x.device)
            model_logits = torch.where(t_broadcast == 0,
                                       pred_x_start_logits,
                                       x_tm1
                                       )

        elif self.model_prediction == 'xprev':
            # Use the logits out of the model directly as the logits for
            # p(x_{t-1}|x_t). model_logits are already set correctly.
            # NOTE: the pred_x_start_logits in this case makes no sense.
            # For Gaussian DDPM diffusion the model predicts the mean of
            # p(x_{t-1}}|x_t), and uses inserts this as the eq for the mean of
            # q(x_{t-1}}|x_t, x_0) to compute the predicted x_0/x_start.
            # The equivalent for the categorical case is nontrivial.
            pred_x_start_logits = model_logits
            raise NotImplementedError(self.model_prediction)

        assert (model_logits.shape ==
                pred_x_start_logits.shape == x.shape + (self.num_classes,))
        return model_logits, pred_x_start_logits 

    # === Sampling ===

    def p_sample(self, model_fn, *, x, t, noise, cond=None, x_logits=None, **kwargs):
        """Sample one timestep from the model p(x_{t-1} | x_t)."""
        model_logits, pred_x_start_logits  = self.p_logits(
            model_fn=model_fn, cond=cond, x=x, t=t, x_logits=x_logits, **kwargs)

        if kwargs.get("guide_step") and int(t[0]) > kwargs.get("guide_step"):
            ref = kwargs.get("ref")
            bs, length = x.shape[0], x.shape[1]
            start = kwargs.get("nth_batch")*kwargs.get("bs")
            ref = torch.LongTensor(ref[start:start+bs]).to(x.device)
            tmp = self.q_sample_for_classifier(ref, t, torch.rand(*model_logits.shape, device=x.device))
            rate = kwargs.get('rate')
            mask = mask_tensor(tmp, model_logits, rate=rate, p=kwargs.get('drop'))  # Dropout Operation
            model_logits = (rate * tmp + (1.-rate) * model_logits)*mask + model_logits*(~mask)

        assert noise.shape == model_logits.shape, noise.shape

        # No noise when t == 0
        # NOTE: for t=0 this just "samples" from the argmax
        #   as opposed to "sampling" from the mean in the gaussian case.
        nonzero_mask = (t != 0).reshape(x.shape[0], *([1] * (len(x.shape))))
        # For numerical precision clip the noise to a minimum value
        noise = torch.clip(noise, min=torch.finfo(noise.dtype).tiny, max=1.)
        gumbel_noise = -torch.log(-torch.log(noise))

        sample = torch.argmax(model_logits + nonzero_mask * gumbel_noise, dim=-1)
        assert sample.shape == x.shape
        assert pred_x_start_logits.shape == model_logits.shape
        return sample, F.softmax(pred_x_start_logits, dim=-1)

    def p_sample_loop_batch(self, num_samples, length, cond=None, batch_size=32, device="cuda", **kwargs):
        num_batch = num_samples // batch_size
        gen_data = []
        # cond = (cond[0].repeat(batch_size,1))
        for b in range(num_batch):
            s = b * batch_size
            if b < num_batch:
                d = self.p_sample_loop(shape=(batch_size, length), cond=cond, device=device, **kwargs)
            else:
                d = self.p_sample_loop(shape=(num_samples - s, length), cond=cond, device=device, **kwargs)

            gen_data.extend(d)
            if kwargs.get("nth_batch"):
                kwargs["nth_batch"] += 1
        gen_data = torch.stack(gen_data, dim=0).to(device).long()
        return gen_data

    def p_sample_loop(self, *, shape, cond=None,
                      num_timesteps=None, return_x_init=False, device="cuda", **kwargs):
        """Ancestral sampling."""
        noise_shape = shape + (self.num_classes,)
        model_fn = self._denoise_fn
        
        def body_fun(i, x, cond=None, x_logits=None):
            bs = shape[0]
            t = torch.full([bs], num_timesteps - 1 - i).to(device)
            if cond:
                cond = cond[:bs]
            
            noise = torch.rand(noise_shape).to(device)
            x, x_logits = self.p_sample(
                model_fn=model_fn,
                x=x,
                t=t,
                cond=cond,
                noise=noise,
                x_logits=x_logits,
                **kwargs,
            )

            return x, x_logits

        if self.transition_mat_type in ['gaussian', 'uniform']:
            # Stationary distribution is a uniform distribution over all pixel values.
            x_init = torch.randint(self.num_classes, shape).to(self.device)

        elif self.transition_mat_type == 'absorbing':
            # Stationary distribution is a kronecker delta distribution
            # with all its mass on the absorbing state.
            # Absorbing state is located at rgb values (128, 128, 128)
            x_init = torch.full(size=shape, fill_value=self.num_classes // 2,
                                dtype=torch.long).to(self.device)
        else:
            raise ValueError(
                f"transition_mat_type must be 'gaussian', 'uniform', 'absorbing' "
                f", but is {self.transition_mat_type}"
            )

        if num_timesteps is None:
            num_timesteps = self.num_timesteps

        final_x = x_init
        x_logits = torch.randn(*(x_init.shape), self.num_classes).to(x_init.device)
        # decodeer = kwargs["decode_fn"]
        for i in range(num_timesteps):
            final_x, x_logits = body_fun(i, final_x, cond=cond,x_logits=x_logits)
        # final_x = jax.lax.fori_loop(lower=0, upper=num_timesteps,
        #                         body_fun=body_fun, init_val=x_init)
            # print(f"{99-i:<3}: {decodeer(final_x)[0]}")
        assert final_x.shape == shape
        if return_x_init:
            return x_init, final_x
        else:
            return final_x

    def vb_terms_bpd(self, model_fn, *, x_start, x_t, t, cond=None):
        """Calculate specified terms of the variational bound.

    Args:
      model_fn: the denoising network
      x_start: original clean data
      x_t: noisy data
      t: timestep of the noisy data (and the corresponding term of the bound
        to return)

    Returns:
      a pair `(kl, pred_start_logits)`, where `kl` are the requested bound terms
      (specified by `t`), and `pred_x_start_logits` is logits of
      the denoised image.
    """
        true_logits = self.q_posterior_logits(x_start, x_t, t, x_start_logits=False)
        model_logits, pred_x_start_logits = self.p_logits(model_fn, cond=cond, x=x_t, t=t)

        kl = categorical_kl_logits(logits1=true_logits, logits2=model_logits)
        # kl = (true_logits.exp() * (true_logits - model_logits)).sum(dim=-1)
        assert kl.shape == x_start.shape
        # uniform 采样
        kl = self.num_timesteps * meanflat(kl) / (np.log(2.)) + self.prior_bpd(x_start=x_start)

        decoder_nll = - categorical_log_likelihood(x_start, model_logits)
        assert decoder_nll.shape == x_start.shape
        decoder_nll = meanflat(decoder_nll) / np.log(2.)

        # print("\n\n", kl, decoder_nll)
        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_start) || p(x_{t-1}|x_t))
        assert kl.shape == decoder_nll.shape == t.shape == (x_start.shape[0],)
        return torch.where(t == 0, decoder_nll, kl), pred_x_start_logits

    def prior_bpd(self, x_start):
        """KL(q(x_{T-1}|x_start)|| U(x_{T-1}|0, self.num_classes-1))."""
        q_probs = self.q_probs(
            x_start=x_start,
            t=torch.full((x_start.shape[0],), self.num_timesteps - 1))

        if self.transition_mat_type in ['gaussian', 'uniform']:
            # Stationary distribution is a uniform distribution over all pixel values.
            prior_probs = torch.ones_like(q_probs) / self.num_classes

        elif self.transition_mat_type == 'absorbing':
            # Stationary distribution is a kronecker delta distribution
            # with all its mass on the absorbing state.
            # Absorbing state is located at rgb values (128, 128, 128)
            absorbing_int = torch.full(size=q_probs.shape[:-1],
                                       fill_value=self.num_classes // 2,
                                       dtype=torch.long)
            prior_probs = F.one_hot(absorbing_int,
                                    num_classes=self.num_classes).float().to(self.device)
        else:
            raise ValueError(
                f"transition_mat_type must be 'gaussian', 'uniform', 'absorbing' "
                f", but is {self.transition_mat_type}"
            )

        assert prior_probs.shape == q_probs.shape

        kl_prior = categorical_kl_probs(
            q_probs, prior_probs)
        assert kl_prior.shape == x_start.shape
        return meanflat(kl_prior) / np.log(2.)

    def cross_entropy_x_start(self, x_start, pred_x_start_logits):
        """Calculate crossentropy between x_start and predicted x_start.

    Args:
      x_start: original clean data
      pred_x_start_logits: predicted_logits

    Returns:
      ce: cross entropy.
    """

        ce = -categorical_log_likelihood(x_start, pred_x_start_logits)
        assert ce.shape == x_start.shape
        ce = meanflat(ce) / np.log(2.)

        assert ce.shape == (x_start.shape[0],)

        return ce

    def training_losses(self, *, x_start, cond=None):
        """Training loss calculation."""

        # Add noise to data
        noise = torch.rand(x_start.shape + (self.num_classes,))
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],)).to(device=x_start.device)

        # t starts at zero. so x_0 is the first noisy datapoint, not the datapoint
        # itself.
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Calculate the loss
        if self.loss_type == 'kl':
            # Optimizes the variational bound L_vb.
            losses, _ = self.vb_terms_bpd(
                model_fn=self._denoise_fn, cond=cond, x_start=x_start, x_t=x_t, t=t)
            # loss_classifier = 0.1 * F.binary_cross_entropy_with_logits(pred_class, cond)
            # losses += loss_classifier

        elif self.loss_type == 'cross_entropy_x_start':
            # Optimizes - sum_x_start x_start log pred_x_start.
            _, pred_x_start_logits = self.p_logits(self._denoise_fn, x=x_t, t=t, cond=cond)
            losses = self.cross_entropy_x_start(
                x_start=x_start, pred_x_start_logits=pred_x_start_logits)

        elif self.loss_type == 'hybrid':
            # Optimizes L_vb - lambda * sum_x_start x_start log pred_x_start.
            vb_losses, pred_x_start_logits = self.vb_terms_bpd(
                model_fn=self._denoise_fn,  cond=cond, x_start=x_start, x_t=x_t, t=t)
            ce_losses = self.cross_entropy_x_start(
                x_start=x_start, pred_x_start_logits=pred_x_start_logits)

            losses = vb_losses + self.hybrid_coeff * ce_losses

        else:
            raise NotImplementedError(self.loss_type)

        assert losses.shape == t.shape
        return losses

    def calc_bpd_loop(self, model_fn, *, x_start, rng):
        """Calculate variational bound (loop over all timesteps and sum)."""
        batch_size = x_start.shape[0]

        noise_shape = x_start.shape + (self.num_classes,)

        def map_fn(map_val):
            t = map_val
            noise = torch.rand(noise_shape)
            # Calculate VB term at the current timestep
            t = torch.full((batch_size,), t)
            vb, _ = self.vb_terms_bpd(
                model_fn=model_fn, x_start=x_start, t=t,
                x_t=self.q_sample(
                    x_start=x_start, t=t,
                    noise=noise))
            assert vb.shape == (batch_size,)
            return vb

        vbterms_tb = map(
            map_fn, (torch.arange(self.num_timesteps)))

        vbterms_bt = torch.tensor(list(vbterms_tb.T))
        assert vbterms_bt.shape == (batch_size, self.num_timesteps)

        prior_b = self.prior_bpd(x_start=x_start)
        total_b = vbterms_tb.sum(dim=0) + prior_b
        assert prior_b.shape == total_b.shape == (batch_size,)

        return {
            'total': total_b,
            'vbterms': vbterms_bt,
            'prior': prior_b,
        }


def mask_tensor(vect, model_logits, rate, p):
    a = torch.rand_like(vect)
    mask = a > p
    # mask = a > rate 
    # model_logits += model_logits *(~mask) / rate
    # return vect * mask, mask
    return mask


