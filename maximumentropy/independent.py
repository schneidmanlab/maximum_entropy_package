from typing import Optional

import numpy as np
import torch
import torch.nn

from . import ArrayOrTensor, OptionalArrayOrTensor
from .base import BaseModel


class Independent(BaseModel):
    def __init__(self, n: int, factors: OptionalArrayOrTensor = None, normalize: bool = False, device: str = 'gpu',
                 tensor_max_size_in_bytes: int = -1, verbose: bool = True) -> None:
        """
        Creates a maximum entropy independent model
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
        if factors is None:
            self.factors = torch.zeros(n, device=self.device)
        else:
            if len(factors) != self.n:
                raise ValueError(f'Wrong number of factors provided. Expected {n}, received {len(factors)}')

            if type(factors) == np.ndarray:
                factors = torch.from_numpy(factors)
            elif type(factors) != torch.Tensor:
                raise TypeError('type(factors) must be torch.Tensor or a numpy.ndarray')

            self.factors = factors.type(torch.get_default_dtype()).to(self.device)

    def get_marginals(self, ps: OptionalArrayOrTensor = None) -> torch.Tensor:
        """
        Calculates the predicted marginals <x_i> of the model.
        :param ps: a vector of 2**model.n probabilities (should be normalized)
        :return: the model predicted marginals <x_i>
        """
        return torch.exp(-self.factors) / (1 + torch.exp(-self.factors))

    def train(self, samples: ArrayOrTensor, **kwargs) -> bool:
        """
        Sets the model factors from the closed form solution, also set the models log_z value
        :param samples: samples to train on
        :return: whether the calculation was successful
        """
        samples = self.move_samples_to_device(samples)
        empirical_marginals = self.get_empirical_marginals(samples)
        n_samples = len(samples)

        # replacing zero and one marginals with 0 or 1
        empirical_marginals[empirical_marginals == 0] = 1 / (2 * n_samples)
        empirical_marginals[empirical_marginals == 1] = 1 - 1 / (2 * n_samples)

        # z = 1 / p(0,0,..,0)
        #   = 1/p(0)p(0)..p(0)
        #   = 1/p(0) * 1/p(0) * 1/p(0) * 1/p(0)
        # log(z) = log(1/p(0) * 1/p(0) * 1/p(0) * 1/p(0))
        #        = log(1/p(0)) + log(1/p(0)) + log(1/p(0)) + log(1/p(0))
        #        = - log(p(0)) - log(p(0)) - log(p(0)) - log(p(0))
        # p(0) = 1 - p(1)
        self.log_z = - torch.log(1 - empirical_marginals).sum()

        # creating a helper matrix, at row i the j'th cell is equal p(x_j=0) = 1-p(x_j=1)
        # if i=j then the cell equals p(x_i=1)
        helper_mat = torch.ones((self.n, self.n), device=self.device)
        helper_mat[range(self.n), range(self.n)] = 2 * empirical_marginals
        helper_mat -= empirical_marginals.repeat(self.n, 1)

        # alpha_i = p_1(0) p_2(0) .. p_i(1) .. p_n(0) = (1-p_1(1))(1-p_2(1))..p_i(1) .. (1-p_n(1)
        self.factors = -torch.log(helper_mat).sum(dim=1) - self.log_z

        return True

    def get_empirical_marginals(self, samples: ArrayOrTensor) -> torch.Tensor:
        """
        Calculates the empirical marginals <x_i> of the given samples.
        :param samples: the empirical samples
        :return: the empirical marginals of the given samples
        """
        samples = self.move_samples_to_device(samples)
        n_samples = samples.shape[0]
        indep_firing_rate = samples.sum(dim=0) / n_samples

        return indep_firing_rate

    def get_energies(self, words: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculates the energies vector of the model for the given words, if words is none than computes the energies
        on all possible words
        E = sum_i alpha_i x_i
        :param words: words to calculate energy on
        :return: energies of the given words
        """
        if words is None:
            self.build_all_words()
            words = self.all_words
        else:
            words = words.to(self.device)

        # calculating energies vector.
        energies = words @ self.factors
        return energies

    def calculate_z(self, **kwargs) -> None:
        """
        Calculates log_z from a closed form solution
        """
        # z = prod (z_i)
        # log(z) = sum(log(z_i))
        # z_i = 1 + exp(-alpha)
        self.log_z = torch.log(1 + torch.exp(-self.factors)).sum()
