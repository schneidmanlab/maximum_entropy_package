import os
import sys
from typing import Optional, Dict, Any, Tuple
import warnings

import torch
import torch.nn

from . import OptionalArrayOrTensor
from .rp import RP


class SigmoidRP(RP):
    def __init__(self, n: int, nprojections: Optional[int] = None, indegree: int = 5, threshold: float = 0.1,
                 projections: OptionalArrayOrTensor = None, projections_thresholds: OptionalArrayOrTensor = None,
                 factors: OptionalArrayOrTensor = None, normalize: bool = False, device: str = 'gpu',
                 tensor_max_size_in_bytes: int = -1, slope: float = 3, verbose: bool = True) -> None:
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
        :param slope: the slope of the sigmoid of the model
        :param verbose: level of printing
        """
        super().__init__(n, nprojections=nprojections, indegree=indegree, threshold=threshold, projections=projections,
                         projections_thresholds=projections_thresholds,
                         factors=factors,
                         normalize=normalize, device=device, tensor_max_size_in_bytes=tensor_max_size_in_bytes,
                         verbose=verbose)
        self.slope = slope

    def apply_projections(self, words: torch.Tensor) -> torch.Tensor:
        """
        Applies the model projections on the given words
        RP_i = sigmoid(slope(sum_j a_ij x_j - theta_i))
        :param words: the words to apply the projections on
        :return: the applied projections
        """
        # one line to consume less memory, important for large models on GPU
        applied_projections = torch.sigmoid(
            self.slope * (self.projections @ words.T - self.projections_thresholds[:, None]))

        return applied_projections

    def reshape(self, samples, iterations: int = 1_000, batch_size: Optional[int] = None, lr: float = 0.1,
                gamma: float = 0.99, nesterov: bool = True, sampler_mat_size: int = 20_000, print_iterations: int = 100,
                save_reshape_cp_filename: Optional[str] = None, save_cp_iterations: int = 100,
                verbose: bool = True
                ) -> None:
        """
        reshape the model using MCMC sampling. evaluate the model marginals at each iteration by generating samples and
        calculate the sampled empirical marginals. Optimizing using SGD (with out without Nesterov momentum)
        :param samples: the train set samples to reshape the model on
        :param iterations: number of iterations to perform
        :param batch_size: batch size, if None then equal to provided set size
        :param lr: learning rate
        :param gamma: the SGD momentum param
        :param nesterov: whether to use SGD with nesterov momentum or not
        :param sampler_mat_size: size of the sampler for the MCMC sample generation step at each iteration
        :param print_iterations: log printing interval
        :param save_reshape_cp_filename: the filename to save checkpoints to
        :param save_cp_iterations: the saving cp interval
        :param verbose: level of printing
        """
        def inner_print(msg: str) -> None:
            if verbose:
                print(msg)
                sys.stdout.flush()

        inner_print(f'Reshaping sigmoidRP model with {self.n} cells and {self.projections.shape[0]} projections on '
                    f'{samples.shape[0]} samples using Markov Chain MonteCarlo sampling and SGD optimizer.')

        if save_reshape_cp_filename is not None:
            save_cp = True
        else:
            save_cp = False
        iteration = 0
        optimizer_state_dict = None

        # loading saved cp if exists
        if save_cp and os.path.isfile(save_reshape_cp_filename):
            print(f'Loading projections and reshape data from saved cp at: {save_reshape_cp_filename}')
            sys.stdout.flush()
            iteration, optimizer_state_dict = self.load_saved_reshape_cp(save_reshape_cp_filename)

        # checking that not all factors are zeros
        if (self.factors == 0).all():
            warnings.warn(f'Model factors are all set to zero, cannot execute reshape.')
            return

        samples = self.move_samples_to_device(samples)
        nsamples = samples.shape[0]

        # creating sparsity mask to optimize only existing synapses
        sparsity_mask = (self.projections != 0).type(torch.get_default_dtype())

        # defining parameters and optimizer
        self.projections = torch.nn.Parameter(self.projections)
        optimizer = torch.optim.SGD([self.projections], lr=lr, momentum=gamma, nesterov=nesterov)
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)

        # defining dataloader
        if batch_size is None:
            batch_size = nsamples
        dataloader = torch.utils.data.DataLoader(samples,
                                                 batch_size=batch_size,
                                                 shuffle=True)

        inner_print(f'Parameters: initial iteration={iteration}, total iterations={iterations}, '
                    f'nsamples={nsamples}, sampler_mat_size={sampler_mat_size}, '
                    f'batch_size={batch_size}, n_batches={len(dataloader)}')
        inner_print(f'torch optimizer: {optimizer}')
        inner_print(f'printing log every {print_iterations} iterations')
        if save_cp:
            inner_print(f'saving checkpoint every {save_cp_iterations} iterations to {save_reshape_cp_filename}')
        inner_print('Reshape start:')

        all_mcmc_samples = []
        while iteration < iterations:
            iteration += 1

            # for each mini batch
            for cur_train_samples in dataloader:
                # generating samples and calculating loss
                with torch.no_grad():
                    mcmc_samples = self.generate_samples(cur_train_samples.shape[0], sampler_mat_size=sampler_mat_size,
                                                         burnin=0)
                empirical_energies = self.get_energies(cur_train_samples)
                mcmc_energies = self.get_energies(mcmc_samples)
                loss = -(mcmc_energies - empirical_energies).mean()

                # Zero gradients, perform a backward pass to calculate the factors gradient
                optimizer.zero_grad()
                loss.backward()

                # keeping only the grad terms of existing synapses using the mask
                self.projections.grad = self.projections.grad * sparsity_mask

                optimizer.step()

                # saving mcmc samples for logging
                if iteration % print_iterations == 0:
                    all_mcmc_samples.append(mcmc_samples)

            if iteration % print_iterations == 0:
                mcmc_samples = torch.cat(all_mcmc_samples)
                all_mcmc_samples = []

                mcmc_marginals = self.get_empirical_marginals(mcmc_samples).detach()
                empirical_marginals = self.get_empirical_marginals(samples).detach()
                empirical_std = self.clopper_pearson_confidence_interval(empirical_marginals, nsamples)

                marginals_delta = mcmc_marginals - empirical_marginals

                normalized_errors = torch.abs(marginals_delta) / empirical_std
                max_error_idx = normalized_errors.argmax()
                max_normalized_error = normalized_errors[max_error_idx]

                inner_print(f'Iter {iteration}: loss={loss.item():.2e}, '
                            f'MSE={(marginals_delta ** 2).mean():.2e}, '
                            f'max error={max_normalized_error:.2f} [{max_error_idx}]; ')

            if save_cp and iteration % save_cp_iterations == 0:
                print(f'Iter {iteration}: saving projections to {save_reshape_cp_filename}')
                sys.stdout.flush()
                self.save_reshape_cp(save_reshape_cp_filename, iteration, optimizer.state_dict())

        self.projections = self.projections.data
        inner_print('Finished reshaping!')

    def save_reshape_cp(self, save_reshape_cp_filename: str, iteration: int,
                        optimizer_state_dict: Dict[str, Any]) -> None:

        data_dict = {
            'iteration': iteration,
            'sampler_state': self.sampler_state.cpu(),
            'optimizer_state_dict': optimizer_state_dict,
            'projections': self.projections.data.cpu().to_sparse(),
            'factors': self.factors.cpu(),
            'projections_thresholds': self.projections_thresholds.cpu()
        }
        torch.save(data_dict, save_reshape_cp_filename)

    def load_saved_reshape_cp(self, save_reshape_cp_filename: str) -> Tuple[int, Dict[str, Any]]:
        try:
            data_dict = torch.load(save_reshape_cp_filename)
        except RuntimeError as e:
            raise (RuntimeError(f'failed to load cp file: {save_reshape_cp_filename}\n'
                                f'original error: {e}').
                   with_traceback(e.__traceback__)) from None

        if 'factors' in data_dict:
            self.factors = data_dict['factors'].type(torch.get_default_dtype()).to(self.device)

        if 'projections' in data_dict:
            self.projections = data_dict['projections'].type(torch.get_default_dtype()).to_dense().to(self.device)

        if 'projections_thresholds' in data_dict:
            self.projections_thresholds = data_dict['projections_thresholds'].type(torch.get_default_dtype()).to(
                self.device)

        if 'sampler_state' in data_dict:
            self.sampler_state = data_dict['sampler_state'].type(torch.get_default_dtype()).to(self.device)

        iteration = data_dict.get('iteration', 0)
        optimizer_state_dict = data_dict.get('optimizer_state_dict')

        return iteration, optimizer_state_dict
