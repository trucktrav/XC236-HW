import numpy as np
import torch
import os
script_directory = os.path.dirname(os.path.abspath(__file__))

if 'solution' in script_directory:
    from solution import utils as ut
    from solution.models import nns
else:
    from submission import utils as ut
    from submission.models import nns

from torch import nn

class GMVAE(nn.Module):
    def __init__(self, nn='v1', z_dim=2, k=500, name='gmvae'):
        super().__init__()
        self.name = name
        self.k = k
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim) # from ./nns/v1.py
        self.dec = nn.Decoder(self.z_dim) # from ./nns/v1.py

        # Mixture of Gaussians prior
        self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k, self.z_dim)
                                        / np.sqrt(self.k * self.z_dim))
        # Uniform weighting
        self.pi = torch.nn.Parameter(torch.ones(k) / k, requires_grad=False)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # To help you start, we have computed the mixture of Gaussians prior
        # prior = (m_mixture, v_mixture) for you, where
        # m_mixture and v_mixture each have shape (1, self.k, self.z_dim)
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be tensor scalars
        #
        # Return:
        #   nelbo, kl_z, rec
        ################################################################################
        # We provide the learnable prior for you. Familiarize yourself with
        # this object by checking its shape.
        prior = ut.gaussian_parameters(self.z_pre, dim=1)
        ### START CODE HERE ###
        nelbo = None
        kl = None
        rec = None

        qm, qv = self.enc(x)
        zs = ut.sample_gaussian(qm, qv)
        pxz = self.dec(zs)

        qln = ut.log_normal(zs, qm, qv)
        qlnm = ut.log_normal_mixture(zs, prior[0], prior[1])

        kl = qln - qlnm
        rec = -1 * ut.log_bernoulli_with_logits(x, pxz)
        # kl = ut.log_mean_exp(kl, dim=-1)
        # rec = ut.log_mean_exp(rec, dim=-1)
        nelbo = torch.mean(kl + rec)
        kl = torch.mean(kl)
        rec = torch.mean(rec)

        return nelbo, kl, rec
        ### END CODE HERE ###
        ################################################################################
        # End of code modification
        ################################################################################
        raise NotImplementedError

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be tensor scalars
        #
        # Return:
        #   niwae, kl, rec
        ################################################################################
        # We provide the learnable prior for you. Familiarize yourself with
        # this object by checking its shape.
        prior = ut.gaussian_parameters(self.z_pre, dim=1)
        ### START CODE HERE ###

        batch = x.shape[0]
        xs = ut.duplicate(x, iw)
        qms, qvs = self.enc(xs)
        zs = ut.sample_gaussian(qms, qvs)
        pxz = self.dec(zs)

        qln = ut.log_normal(zs, qms, qvs)
        pms = prior[0]
        pvs = prior[1]
        qlnm = ut.log_normal_mixture(zs, pms, pvs)

        kls = qln - qlnm
        recs = ut.log_bernoulli_with_logits(xs, pxz)
        logpt = recs - kls
        temp = ut.log_mean_exp(logpt.reshape(iw, batch), 0)
        niwae = -1.0 * torch.mean(temp)

        kl = torch.mean(kls)
        rec = -1 * torch.mean(recs)

        return niwae, kl, rec
        ### END CODE HERE ###
        ################################################################################
        # End of code modification
        ################################################################################
        raise NotImplementedError

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        m, v = ut.gaussian_parameters(self.z_pre.squeeze(0), dim=0)
        idx = torch.distributions.categorical.Categorical(self.pi).sample((batch,))
        m, v = m[idx], v[idx]
        return ut.sample_gaussian(m, v)

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
