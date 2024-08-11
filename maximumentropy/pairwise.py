import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn

from . import ArrayOrTensor, OptionalArrayOrTensor
from .base import BaseModel


class Pairwise(BaseModel):
    def __init__(self, n: int, factors: OptionalArrayOrTensor = None, normalize: bool = False, device: str = 'gpu',
                 tensor_max_size_in_bytes: int = -1, verbose: bool = True) -> None:
        """
        Creates a maximum entropy pairwise model
        :param n: size of the model, number of cells
        :param factors: the models factors, if not provided initialized with zeros
        :param normalize: should normalize the model
        :param device: cpu or gpu, if gpu is available then uses gpu unless specified otherwise
        :param tensor_max_size_in_bytes:  max size of tensor in bytes, used to avoid CUDA out of mem error,
        if not specified then it is calculated based on available GPU mem
        :param verbose: level of printing
        """
        super().__init__(n, normalize=normalize, device=device, tensor_max_size_in_bytes=tensor_max_size_in_bytes,
                         verbose=verbose)
        n_factors = n * (n + 1) // 2

        if factors is None:
            self.factors = torch.zeros(n_factors, device=self.device)
        else:
            if len(factors) != n_factors:
                raise ValueError(f'Wrong number of factors provided. Expected {n_factors}, received {len(factors)}')
            if type(factors) == np.ndarray:
                factors = torch.from_numpy(factors)
            elif type(factors) != torch.Tensor and type(factors) != torch.nn.Parameter:
                raise TypeError('type(factors) must be torch.Tensor, torch.nn.Parameter or a numpy.ndarray')

            self.factors = factors.type(torch.get_default_dtype()).to(self.device)

            if normalize:
                # calculate z
                self.calculate_z()

    def build_factors_mat(self) -> torch.Tensor:
        """
        creates a factors matrix from the 1-d model.factors. first n factors are on the diagonal (alpha_i),
        the other n(n-1)/2 factors (beta_ij) are the upper triangle. the lower triangle is set to zero
        :return: the factors matrix
        """
        alphas = self.factors[:self.n]
        betas = self.factors[self.n:]
        triu_indices = torch.triu_indices(self.n, self.n, 1)
        factors_mat = torch.diag(alphas)
        factors_mat[triu_indices[0], triu_indices[1]] = betas
        return factors_mat

    def get_marginals(self, ps: torch.Tensor = None) -> torch.Tensor:
        """
        Calculates the marginals <x_i> and <x_i x_j> of the model.
        :param ps: a vector of 2**model.n probabilities (should be normalized)
        :return: the model predicted marginals. First n elements are <x_i> values. Last n(n-1)/2 are  <x_i x_j> values.
        """
        if self.n > self.MAX_CELLS_FOR_EXHAUSTIVE:
            warnings.warn(f'Calculating model marginals exhaustively for {self.n} cells.')

        if ps is None:
            ps = self.get_probability()
        else:
            self.build_all_words()
            ps = ps.to(self.device)
        words = self.all_words
        firing_rates_mat = words.T @ (words * ps[:, None])

        # getting <x_i>
        indep_firing_rates = firing_rates_mat.diagonal(0)

        # getting <x_i x_j>
        triu_indices = torch.triu_indices(self.n, self.n, 1)
        pair_firing_rates = firing_rates_mat[triu_indices[0], triu_indices[1]]

        # concatenating independent and pair firing rates
        marginals = torch.cat((indep_firing_rates, pair_firing_rates))

        return marginals

    def get_energies(self, words: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculates the energies vector of the model for the given words, if words is none than computes the energies
        on all possible words
        E = sum_i alpha_i x_i + sum_i<j beta_ij x_i x_j
        :param words: words to calculate energy on
        :return: energies of the given words
        """
        if words is None:
            self.build_all_words()
            words = self.all_words
        else:
            words = words.to(self.device)
        factors_mat = self.build_factors_mat()

        # calculating energies vector.
        energies = ((words @ factors_mat) * words).sum(1)
        return energies

    def get_empirical_marginals(self, samples: ArrayOrTensor) -> torch.Tensor:
        """
        Calculates the empirical marginals <x_i> and <x_i x_j> of the given samples.
        :param samples: the empirical samples
        :return: the empirical marginals of the given samples
        """
        samples = self.move_samples_to_device(samples)
        n_samples = samples.shape[0]
        samples_mat_mul = samples.T @ samples / n_samples

        indep_firing_rate = samples_mat_mul.diagonal(0)
        triu_indices = torch.triu_indices(self.n, self.n, 1)
        co_firing_rate = samples_mat_mul[triu_indices[0], triu_indices[1]]

        empirical_marginals = torch.cat((indep_firing_rate, co_firing_rate))
        return empirical_marginals
