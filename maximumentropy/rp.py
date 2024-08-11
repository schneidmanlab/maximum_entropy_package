import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn

from . import ArrayOrTensor, OptionalArrayOrTensor
from .base import BaseModel


class RP(BaseModel):
    def __init__(self, n: int, nprojections: Optional[int] = None, indegree: int = 5, threshold: float = 0.1,
                 projections: OptionalArrayOrTensor = None, projections_thresholds: OptionalArrayOrTensor = None,
                 factors: OptionalArrayOrTensor = None, normalize: bool = False, device: str = 'gpu',
                 tensor_max_size_in_bytes: int = -1, verbose: bool = True) -> None:
        """
        Creates a maximum entropy random projections model
        :param n: size of the model, number of cells
        :param nprojections: number of projections in the model, default value is n(n+1)/2.
        ignored if projections are provided
        :param indegree: The average in-degree of each projection. ignored if projections are provided
        :param threshold: the threshold value of each projection (multiplied by indegree).
        ignored if projections are provided
        :param projections: The projections of the model, if provided then must also provide projections thresholds.
        :param projections_thresholds: The threshold of each projection.
        :param factors: the models factors, if not provided initialized with zeros
        :param normalize: should normalize the model
        :param device: cpu or gpu, if gpu is available then uses gpu unless specified otherwise
        if not specified then it is calculated based on available GPU mem
        :param tensor_max_size_in_bytes:  max size of tensor in bytes, used to avoid CUDA out of mem error,
        :param verbose: level of printing
        """
        super().__init__(n, normalize=normalize, device=device, tensor_max_size_in_bytes=tensor_max_size_in_bytes,
                         verbose=verbose)
        if projections is not None:
            if projections_thresholds is None:
                raise ValueError('projections_thresholds vector must be provided with projections.')

            if type(projections) == np.ndarray:
                self.projections = torch.from_numpy(projections).type(torch.get_default_dtype())
            elif type(projections) == torch.Tensor:
                self.projections = projections.type(torch.get_default_dtype())
            else:
                raise TypeError('type(projections) must be torch.Tensor or numpy.ndarray')

            if type(projections_thresholds) == np.ndarray:
                self.projections_thresholds = torch.from_numpy(projections_thresholds).type(torch.get_default_dtype())
            elif type(projections_thresholds) == torch.Tensor:
                self.projections_thresholds = projections_thresholds.type(torch.get_default_dtype())
            else:
                raise TypeError('type(projections_thresholds) must be torch.Tensor or numpy.ndarray')

            if self.projections.shape[0] != self.projections_thresholds.shape[0]:
                raise ValueError('projections first dimension must be equal to projections_thresholds first dimension')

            if self.projections.shape[1] != self.n:
                raise ValueError('projections second dimension must be equal to the number of cells n.')

            self.projections = self.projections.to(self.device)
            self.projections_thresholds = self.projections_thresholds.to(self.device)

            self.n_projections = self.projections.shape[0]
            self.n_factors = self.n_projections

            if factors is None:
                self.factors = torch.zeros(self.n_factors, device=self.device)
            else:
                if len(factors) != self.n_factors:
                    raise ValueError(
                        'Wrong number of factors provided. Expected {}, received {}'.format(self.n_factors,
                                                                                            len(factors)))

                if type(factors) == np.ndarray:
                    factors = torch.from_numpy(factors).type(torch.get_default_dtype())
                elif type(factors) != torch.Tensor and type(factors) != torch.nn.Parameter:
                    raise TypeError('type(factors) must be torch.Tensor, torch.nn.Parameter or a numpy.ndarray')

                self.factors = factors.type(torch.get_default_dtype()).to(self.device)
        else:
            if nprojections is None:
                # default - same number of factors as ising
                self.n_projections = n * (n + 1) // 2
            elif type(nprojections) == int:
                self.n_projections = nprojections
            else:
                raise TypeError('nprojections must be an integer')

            self.projections_thresholds = torch.ones(self.n_projections, device=self.device) * threshold * indegree

            if factors is not None:
                raise ValueError(
                    'The variable "factors" has no meaning if the projections were not set as well')

            self.n_factors = self.n_projections
            self.factors = torch.zeros(self.n_factors, device=self.device)

            self.projections = self.generate_projections(indegree).to(self.device)
        if normalize:
            # calculate z
            self.calculate_z()

        self._n_projections = self.n_projections
        self._n_factors = self.n_factors

    def generate_projections(self, indegree: int) -> torch.Tensor:
        """
        generate a new set of random projections
        :param indegree: the average in degree of each projection
        :return: a new set of random projections
        """
        # create random sparse connections
        dist = torch.distributions.Normal(1, 1)
        projections = dist.sample((self.n_projections, self.n))
        if indegree < self.n:
            sparsity = indegree / self.n
            inactive_projections = torch.rand((self.n_projections, self.n)) > sparsity
            projections[inactive_projections] = 0
        return projections

    def apply_projections(self, words: torch.Tensor) -> torch.Tensor:
        """
        Applies the model projections on the given words
        RP_i = Theta(sum_j a_ij x_j - theta_i)
        :param words: the words to apply the projections on
        :return: the applied projections
        """
        # one line to consume less memory, important for large models on GPU
        applied_projections = (self.projections @ words.T > self.projections_thresholds[:, None]).type(
            torch.get_default_dtype())
        return applied_projections

    def get_marginals(self, ps: torch.Tensor = None) -> torch.Tensor:
        """
        Calculates the marginals <RP_i> of the model.
        :param ps: a vector of 2**model.n probabilities (should be normalized)
        :return: the model predicted marginals <RP_i>.
        """
        if self.n > self.MAX_CELLS_FOR_EXHAUSTIVE:
            warnings.warn(f'Calculating model marginals exhaustively for {self.n} cells.')

        if ps is None:
            ps = self.get_probability()
        else:
            self.build_all_words()
            ps = ps.to(self.device)
        words = self.all_words

        n_chunks = self.get_applied_projections_n_chunks(words.shape[0])

        marginals = torch.zeros(self.n_projections, device=self.device)
        for cur_ps, cur_words in zip(ps.chunk(n_chunks), words.chunk(n_chunks)):
            # The (i,j) entry is the i'th RP calculated for the j'th sample
            applied_projections = self.apply_projections(cur_words)
            marginals += applied_projections @ cur_ps

        return marginals

    def get_applied_projections_n_chunks(self, n_samples: int) -> int:
        """
        Calculates the number of chunks to execute apply projection. In large models we might get CUDA out of mem error
        so chunks are needed
        :param n_samples: number of samples we want to apply the projections on
        :return: number of chunks
        """
        if self.tensor_max_size_in_bytes == -1:
            n_chunks = 1
        else:
            max_samples = int(self.tensor_max_size_in_bytes / self.n_projections / self.projections.element_size())
            n_chunks = n_samples // max_samples + 1
        return n_chunks

    def get_energies(self, words: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculates the energies vector of the model for the given words, if words is none than computes the energies
        on all possible words
        E = sum_i lambda_i RP_i
        :param words: words to calculate energy on
        :return: energies of the given words
        """
        if words is None:
            self.build_all_words()
            words = self.all_words
        else:
            words = words.to(self.device)

        # splitting samples into chunks to avoid out of memory errors if applied_projections tensor would be too large
        # if tensor max size is -1 then not splitting at all
        n_chunks = self.get_applied_projections_n_chunks(words.shape[0])

        energies = []
        for cur_chunk in words.chunk(n_chunks):
            applied_projections = self.apply_projections(cur_chunk)
            energies.append(self.factors @ applied_projections)

        return torch.cat(energies)

    def get_empirical_marginals(self, samples: ArrayOrTensor) -> torch.Tensor:
        """
        Calculates the empirical marginals <RP_i> of the given samples.
        :param samples: the empirical samples
        :return: the empirical marginals of the given samples
        """
        samples = self.move_samples_to_device(samples)

        # splitting samples into chunks to avoid out of memory errors if applied_projections tensor would be too
        # if tensor max size is -1 then not splitting at all
        n_chunks = self.get_applied_projections_n_chunks(samples.shape[0])

        applied_projections = torch.zeros(self.n_projections, device=self.device)
        for cur_chunk in samples.chunk(n_chunks):
            # The (i,j) entry is the i'th RP calculated for the j'th sample
            applied_projections += self.apply_projections(cur_chunk).sum(1)

        return applied_projections / samples.shape[0]
