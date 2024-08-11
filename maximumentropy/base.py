import itertools as it
import math
import os
import sys
import time
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, Dict, Any

import numpy as np
import scipy.stats
import scipy.stats as stats
import torch
import torch.nn

from . import ArrayOrTensor

torch.set_default_dtype(torch.float32)


class BaseModel(ABC):
    def __init__(self, n: int, normalize: bool = False, device: str = 'gpu', tensor_max_size_in_bytes: int = -1,
                 verbose: bool = True) -> None:
        """
        Initialize a maximum entropy model
        :param n: size of the model, number of cells
        :param normalize: should normalize the model
        :param device: cpu or gpu, if gpu is available then uses gpu unless specified otherwise
        :param tensor_max_size_in_bytes:  max size of tensor in bytes, used to avoid CUDA out of mem error,
        if not specified then it is calculated based on available GPU mem
        :param verbose: level of printing
        """
        self.n = n
        self.all_words = None
        self.log_z = None
        self.factors = None
        self.sampler_state = None
        self.MAX_CELLS_FOR_EXHAUSTIVE = 20

        if isinstance(device, torch.device):
            self.device = device
        elif device.lower() == 'cpu' or not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda')

        if verbose:
            print(f'Created {type(self).__name__} maximum entropy model for {self.n} cells, '
                  f'using {self.device} for pytorch operations.')
            sys.stdout.flush()

        # setting tensor max size to avoid out of memory errors.
        # if device is cpu then default is -1, i.e. no mem limit
        # if device is gpu then checking how much memory is available on the device in order to set max tensor size
        if self.device.type == 'cuda' and tensor_max_size_in_bytes == -1:
            cuda_total_memory_in_bytes = torch.cuda.get_device_properties(self.device).total_memory

            # leaving half of the memory + 1gb available
            self.tensor_max_size_in_bytes = cuda_total_memory_in_bytes // 2 - 2 ** 30

            if verbose:
                print(f'total cuda available memory:{int(cuda_total_memory_in_bytes / 2 ** 30) + 1}GB, '
                      f'model max tensor size={int(self.tensor_max_size_in_bytes / 2 ** 30) + 1}GB')
        else:
            self.tensor_max_size_in_bytes = tensor_max_size_in_bytes
            if verbose and self.device.type == 'cuda':
                cuda_total_memory_in_bytes = torch.cuda.get_device_properties(self.device).total_memory
                print(f'total cuda available memory:{int(cuda_total_memory_in_bytes / 2 ** 30) + 1}GB, '
                      f'model max tensor size={int(self.tensor_max_size_in_bytes / 2 ** 30) + 1}GB')
        sys.stdout.flush()

        if normalize:
            # build all words
            self.build_all_words()

    # Methods Implemented by BaseModel
    def build_all_words(self) -> None:
        """
        build all possible 2**model.n binary words and save in model.all_words
        """
        if self.all_words is None:
            if self.n > self.MAX_CELLS_FOR_EXHAUSTIVE:
                warnings.warn(f'Building all_words for {self.n} cells.')

            self.all_words = torch.tensor(list(it.product([0, 1], repeat=self.n)),
                                          device=self.device, dtype=torch.get_default_dtype()).fliplr()

    def train(self, samples: ArrayOrTensor, force_exhaustive: bool = False, force_MCMC: bool = False,
              verbose: bool = True, **kwargs) -> Union[bool, Tuple[bool, int]]:
        """
        train the model either exhaustively or using MCMC
        :param samples: the train set samples to train the model on
        :param force_exhaustive: force exhaustive training
        :param force_MCMC: force MCMC training
        :param verbose: level of printing
        :param kwargs: additional params for the exhaustive and MCMC training functions
        :return: whether the model converged or not, and the convergence iteration (if in benchmarking mode)
        """
        if force_exhaustive:
            return self.train_model_exhaustive(samples, verbose=verbose, **kwargs)
        elif force_MCMC or self.n > self.MAX_CELLS_FOR_EXHAUSTIVE:
            return self.train_model_mcmc(samples, verbose=verbose, **kwargs)
        else:
            return self.train_model_exhaustive(samples, verbose=verbose, **kwargs)

    def train_model_mcmc(self, samples: ArrayOrTensor, pval: float = 0.1, max_steps: int = 1e10,
                         gradient_memory_size=30, lr: float = 0.1,
                         gamma: float = 0.99, nesterov: bool = True, sampler_mat_size: int = 20_000,
                         max_nsamples_update_interval: int = 5_000,
                         save_cp_filename: Optional[str] = None, save_cp_iterations: int = 100,
                         save_cp_append: bool = False,
                         verbose: bool = True, benchmarking: bool = False) -> Union[bool, Tuple[bool, int]]:
        """
        train the model using MCMC sampling. evaluate the model marginals at each iteration by generating samples and
        calculate the sampled empirical marginals.
        Optimizing using SGD, lr decays like 1/sqrt(t) and normalized by the gradient norm
        (averaged over gradient_memory_size steps).
        Model converges after all marginals are below the threshold value for gradient_memory_size consecutive steps.
        The convergence threshold is calculated using the given pval. What are the chances that the marginals of the
        wrong model would within threshold standard deviations from the empirical marginals for gradient_memory_size
        consecutive steps.
        :param samples: the train set samples to train the model on
        :param pval: the pval to calculate the threshold from
        :param max_steps: max number of training iterations
        :param lr: initial learning rate
        :param gradient_memory_size: size of the gradient memory, used to calculate the convergence threshold and
        the learning rate
        :param gamma: the SGD momentum param
        :param nesterov: whether to use SGD with nesterov momentum or not
        :param sampler_mat_size: size of the sampler for the MCMC sample generation step at each iteration
        :param max_nsamples_update_interval: after how many iteration to increase the number of MCMC samples generated
        :param save_cp_filename: the filename to save checkpoints to
        :param save_cp_iterations: the saving cp interval
        :param save_cp_append: whether to append the trained factors values when saving or to override
        :param verbose: level of printing
        :param benchmarking: benchmarking or not, if true returns the converging iteration
        :return: whether the model converged or not, and the convergence iteration (if in benchmarking mode)
        """

        def inner_print(msg: str) -> None:
            if verbose:
                print(msg)
                sys.stdout.flush()

        inner_print(f'Pval training {self.n} cells on {samples.shape[0]} samples '
                    f'using Markov Chain MonteCarlo sampling and SGD optimizer.')
        nsamples = samples.shape[0]
        orig_nsamples = math.ceil(nsamples / sampler_mat_size) * sampler_mat_size
        samples = self.move_samples_to_device(samples)

        # compute confidence interval for the empirical measurements, using one std range (hence alpha=0.32)
        empirical_marginals = self.get_empirical_marginals(samples)
        empirical_std = self.get_empirical_std(samples)
        n_marginals = empirical_std.shape[0]

        def get_threshold_from_pval(pval: float, n_marginals: int, mem_size: int) -> float:
            p = 1 - math.pow(pval, 1 / (n_marginals * mem_size))
            threshold = np.abs(stats.norm.ppf(p / 2))
            return threshold

        zero_marginal_threshold = get_threshold_from_pval(pval, n_marginals, gradient_memory_size)

        if save_cp_filename is not None:
            save_cp = True
        else:
            save_cp = False

        iteration = 0
        gradient_memory = None
        max_normalized_error = torch.inf
        max_error_idx = -1
        optimizer_state_dict = None

        # loading saved cp if exists
        if save_cp and os.path.isfile(save_cp_filename):
            inner_print(f'Loading training data and factors from saved cp at: {save_cp_filename}')
            iteration, gradient_memory, optimizer_state_dict = self.load_saved_cp(
                save_cp_filename)

        if gradient_memory is None:
            # building gradient memory, initializing with empirical_std times 100
            gradient_memory = empirical_std.repeat(gradient_memory_size, 1) * 100
        else:
            gradient_memory = gradient_memory.type(torch.get_default_dtype()).to(self.device)

        gradient_norm_memory = (gradient_memory * gradient_memory).sum(1)

        self.factors = torch.nn.Parameter(self.factors)
        optimizer = torch.optim.SGD([self.factors], lr=lr, momentum=gamma, nesterov=nesterov)

        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)

        # learning rate goes like 1/sqrt(iteration) and is normalized by gradient norm
        def get_lr(epoch: int) -> float:
            if epoch == 0:
                return 1
            else:
                g2 = gradient_norm_memory.mean()
                return 1 / torch.sqrt(g2 * epoch)

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr, last_epoch=iteration - 1)

        inner_print(f'Parameters: initial iteration={iteration}, '
                    f'initial nsamples={nsamples}, '
                    f'gradient memory={gradient_memory_size}, sampler_mat_size={sampler_mat_size}, '
                    f'max_steps={max_steps:.2e}, device={self.device}, pval={pval}')
        inner_print(
            f'training converges when cur_max_error for {n_marginals} marginals is below {zero_marginal_threshold:.2f} '
            f'for {gradient_memory_size} consecutive steps.')
        inner_print(f'torch optimizer: {optimizer}')
        inner_print(f'multiplying lr at each step by 1/(|grad| * sqrt(t))')
        if save_cp:
            inner_print(f'saving checkpoint every {save_cp_iterations} iterations to {save_cp_filename}')
        inner_print('Train start:')

        last_print = time.time()
        converged = False
        converged_iterations = 0
        print_msg = ''
        nsamples_to_generate = ((iteration - 1) // max_nsamples_update_interval + 1) * orig_nsamples

        while not converged and iteration < max_steps:
            iteration += 1

            # calculating number of samples to generate
            nsamples_to_generate = ((iteration - 1) // max_nsamples_update_interval + 1) * orig_nsamples

            # generating samples (without gradient)
            with torch.no_grad():
                mcmc_samples = self.generate_samples(nsamples_to_generate, sampler_mat_size=sampler_mat_size, burnin=0)
            # calculating generated samples empirical marginals
            mcmc_marginals = self.get_empirical_marginals(mcmc_samples)

            # calculating loss to get desired gradient
            loss = -((mcmc_marginals - empirical_marginals) * self.factors).sum()

            # Zero gradients, perform a backward pass to calculate the factors gradient
            optimizer.zero_grad()
            loss.backward()
            gradient = self.factors.grad

            # calculating normalized error
            gradient_memory[iteration % gradient_memory_size, :] = gradient
            normalized_errors = torch.sqrt((gradient_memory ** 2).mean(0)) / empirical_std
            max_normalized_error, max_error_idx = normalized_errors.max(0)

            gradient_norm_memory[iteration % gradient_memory_size] = (gradient * gradient).sum()

            # Checking if cur norm errors are below thresholds
            cur_norm_errors = gradient.abs() / empirical_std
            cur_max_normalized_error, cur_max_error_idx = cur_norm_errors.max(0)
            if cur_max_normalized_error > zero_marginal_threshold:
                converged_iterations = 0
                print_msg = (f'cur max error {cur_max_normalized_error:.2f} [{cur_max_error_idx}] > '
                             f'{zero_marginal_threshold:.2f}')
            else:
                converged_iterations += 1
                print_msg = (f'cur max error {cur_max_normalized_error:.2f} [{cur_max_error_idx}]. '
                             f'{converged_iterations} iterations converged')

            # printing state every three seconds
            if time.time() - last_print > 3:
                last_print = time.time()

                inner_print(
                    f'Iter {iteration}: samples={nsamples_to_generate}, lr={lr_scheduler.get_last_lr()[0]:.2e}, '
                    f'MSE={(gradient_memory ** 2).mean():.2e}, '
                    f'max error={max_normalized_error:.2f} [{max_error_idx}]; '
                    f'{print_msg}')

            # saving
            if save_cp and iteration % save_cp_iterations == 0:
                inner_print(f'Iteration {iteration}: Saving checkpoint to {save_cp_filename}')
                self.save_cp(save_cp_filename, iteration, gradient_memory, optimizer.state_dict(),
                             save_cp_append=save_cp_append)

            # Checking if converged
            converged = (converged_iterations >= gradient_memory_size)
            if not converged:
                # taking an optimization step, updating learning rate
                optimizer.step()
                lr_scheduler.step()

        inner_print(
            f'Iter {iteration}: samples={nsamples_to_generate}, lr={lr_scheduler.get_last_lr()[0]:.2e}, '
            f'MSE={(gradient_memory ** 2).mean():.2e}, '
            f'max error={max_normalized_error:.2f} [{max_error_idx}]; '
            f'{print_msg}')

        if converged:
            inner_print(f'Converged! finished training in {iteration} iterations')
        else:
            inner_print(f'Did not converge after {iteration} iterations. Stopping the training process.')

        self.factors = self.factors.data
        self.log_z = None

        if benchmarking:
            return bool(converged), iteration
        else:
            return bool(converged)

    def train_model_exhaustive(self, samples: ArrayOrTensor, threshold: float = 1.3, max_steps: int = 1e10,
                               lr: float = 0.1, decay_factor: float = 0.75, min_lr: float = 1e-7,
                               decay_patience: int = 30, verbose: bool = True, benchmarking: bool = False) -> Union[
        bool, Tuple[bool, int]]:
        """
        train the model exhaustively. at each iteration calculates the model marginals exhaustively.
        Optimizing using torch.optimizer.Adam with ReduceLROnPlateau learning rate scheduler.
        Model converges when the predicted marginals are within threshold standard deviations from the empirical marginals.
        :param samples: the train set samples to train the model on
        :param threshold: the convergence threshold. the model converges when all predicted marginals are within
        threshold standard deviations from the corresponding empirical marginal
        :param max_steps: max number of training iterations
        :param lr: initial learning rate
        :param decay_factor: the lr decay factor to be used by the ReduceLROnPlateau learning rate scheduler
        :param min_lr: min learning rate for ReduceLROnPlateau learning rate scheduler
        :param decay_patience: patience for ReduceLROnPlateau learning rate scheduler
        :param verbose: level of printing
        :param benchmarking: benchmarking or not, if true returns the converging iteration
        :return: whether the model converged or not, and the convergence iteration (if in benchmarking mode)
        """

        # creating network model to be trained
        def inner_print(msg: str) -> None:
            if verbose:
                print(msg)
                sys.stdout.flush()

        if self.n > self.MAX_CELLS_FOR_EXHAUSTIVE:
            warnings.warn(f'Training model exhaustively for {self.n} cells is not recommended, use MCMC training.')
        inner_print(f'Training Exhaustively {self.n} cells.')

        samples = self.move_samples_to_device(samples)

        self.factors = torch.nn.Parameter(self.factors)
        # factors = self.factors
        optimizer = torch.optim.Adam([self.factors], lr=lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=decay_factor,
                                                                  patience=decay_patience,
                                                                  cooldown=decay_patience,
                                                                  threshold=0,
                                                                  min_lr=min_lr,
                                                                  verbose=verbose)

        empirical_marginals = self.get_empirical_marginals(samples).to(self.device)
        empirical_std = self.get_empirical_std(samples)

        self.build_all_words()

        iteration = 0
        last_print = time.time()

        inner_print(f'Training Exhaustively {self.n} cells.')
        inner_print(f'Parameters: threshold={threshold}, lr = {lr}, max_steps={max_steps}, device={self.device}')
        inner_print(f'torch optimizer: {optimizer}')
        inner_print(f'torch lr_scheduler: {lr_scheduler.state_dict()}')
        inner_print('Train start:')

        converged = False
        while not converged and iteration < max_steps:
            iteration += 1

            # Forward pass: Compute predicted marginals.
            self.log_z = None
            with torch.no_grad():
                predicted_marginals = self.get_marginals()

            # the gradient of the factors is the delta between predicted and empirical marginals
            loss = -((predicted_marginals - empirical_marginals) * self.factors).sum()

            # Zero gradients, perform a backward pass to calculate the factors gradient
            optimizer.zero_grad()
            loss.backward()
            gradient = self.factors.grad

            # # calculating normalized error
            normalized_errors = torch.abs(predicted_marginals - empirical_marginals) / empirical_std
            max_normalized_error = normalized_errors.max()

            # printing state every three seconds
            if time.time() - last_print > 3:
                last_print = time.time()
                n_converged_marginals = (normalized_errors < threshold).sum()
                inner_print(f'Iteration {iteration}: MSE= {(gradient ** 2).mean():.3e}, '
                            f'max normalized error={max_normalized_error:.3f} '
                            f'{n_converged_marginals}/{self.factors.shape[0]} marginals converged')

            # Checking if converged
            if max_normalized_error < threshold:
                converged = True
            else:
                optimizer.step()
                lr_scheduler.step((gradient ** 2).mean())

        n_converged_marginals = (normalized_errors < threshold).sum()
        inner_print(f'Iteration {iteration}: MSE= {(gradient ** 2).mean():.3e}, '
                    f'max normalized error={max_normalized_error:.3f} '
                    f'{n_converged_marginals}/{self.factors.shape[0]} marginals converged')

        if converged:
            inner_print(f'Converged! finished training in {iteration} iterations')
        else:
            inner_print(f'Did not converge after {iteration} iterations. Stopping the training process.')

        self.factors = self.factors.data
        self.log_z = None

        if benchmarking:
            return bool(converged), iteration
        else:
            return bool(converged)

    def load_saved_cp(self, save_cp_filename: str) -> Tuple[int, torch.Tensor, Dict[str, Any]]:
        """
        Load a previously saved cp file during training. sets the model.factors and the models.sampler_state
        according to the values in the saves file
        :param save_cp_filename: the saved cp filename to load
        :return: the saved iteration, gradient memory, and torch.optimizer state_dict
        """
        try:
            data_dict = torch.load(save_cp_filename)
        except RuntimeError as e:
            raise (RuntimeError(f'failed to load cp file: {save_cp_filename}\n'
                                f'original error: {e}').
                   with_traceback(e.__traceback__)) from None

        if 'factors' in data_dict:
            if type(data_dict['factors']) == list:
                factors = data_dict['factors'][-1][1]
            else:
                factors = data_dict['factors']
            self.factors = factors.type(torch.get_default_dtype()).to(self.device)

        if 'sampler_state' in data_dict:
            self.sampler_state = data_dict['sampler_state'].type(torch.get_default_dtype()).to(self.device)

        gradient_memory = data_dict.get('gradient_memory')
        iteration = data_dict.get('iteration', 0)
        optimizer_state_dict = data_dict.get('optimizer_state_dict')

        return iteration, gradient_memory, optimizer_state_dict

    def save_cp(self, save_cp_filename: str, iteration: int, gradient_memory: torch.Tensor,
                optimizer_state_dict: Dict[str, Any],
                save_cp_append: bool = False) -> None:
        """
        Saves a model checkpoint during training.
        :param save_cp_filename: the filename to save to
        :param iteration: the training iteration
        :param gradient_memory: the gradient memory
        :param optimizer_state_dict: the torch.optimizer state_dict
        :param save_cp_append: whether to append the current model state to the cp file or to override and save
        only the latest value
        """
        if save_cp_append:
            if os.path.isfile(save_cp_filename):
                data_dict = torch.load(save_cp_filename)
            else:
                data_dict = {}

            if 'factors' in data_dict and type(data_dict['factors']) == list:
                data_dict['factors'].append((iteration, self.factors.data.cpu()))
            else:
                data_dict['factors'] = [(iteration, self.factors.data.cpu())]

        else:
            data_dict = {'factors': self.factors.data.cpu()}

        data_dict['sampler_state'] = self.sampler_state.cpu()
        data_dict['gradient_memory'] = gradient_memory.cpu()
        data_dict['iteration'] = iteration
        data_dict['optimizer_state_dict'] = optimizer_state_dict

        torch.save(data_dict, save_cp_filename)

    def calculate_z(self, force_exhaustive=False, K=5_000, n_beta=10_000) -> None:
        """
        Calculates the partition function of the model. If it is a small model (n <= model.MAX_CELLS_FOR_EXHAUSTIVE)
        the calculation is exhaustive, otherwise the partition function is approximated using the annealed importance
        sampling method. see section 18.7.1 in The Deep Learning book:
        https://www.deeplearningbook.org/contents/partition.html
        When using annealed importance sampling default K, n_beta values are ok for small models (n<100),
        for larger models n_beta needs to be larger.
        :param force_exhaustive: should force exhaustive calculation even when n > model.MAX_CELLS_FOR_EXHAUSTIVE.
        :param K: Number of random states in the model to do mcmc step on each intermediate distribution.
        :param n_beta: number of intermediate probability distributions between uniform and the models distribution.
        """
        if force_exhaustive or self.n <= self.MAX_CELLS_FOR_EXHAUSTIVE:
            if self.n > self.MAX_CELLS_FOR_EXHAUSTIVE:
                warnings.warn(f'Calculating z exhaustively for {self.n} cells.')

            ps = torch.exp(-self.get_energies())
            z = ps.sum()
            self.log_z = torch.log(z)
        else:
            self.log_z = self.annealed_importance_sampling(K=K, n_beta=n_beta)

    def annealed_importance_sampling(self, K: int = 5_000, n_beta: int = 10_000) -> torch.Tensor:
        """
        Calculate an approximation for the partition function using the annealed importance sampling method.
        Section 18.7.1 in The Deep Learning book:
        https://www.deeplearningbook.org/contents/partition.html
        default K, n_beta values are ok for small models (n<100), for larger models n_beta needs to be larger.
        :param K: Number of random states in the model to do mcmc step on each intermediate distribution.
        :param n_beta: number of intermediate probability distributions between uniform and the models distribution.
        :return: log(z) approximation
        """
        # K is the number of random states in the model
        # n_beta is the number of steps from the known distribution to the target one
        samples = torch.randint(0, 2, size=(K, self.n), device=self.device).type(torch.get_default_dtype())
        current_energies = torch.zeros(K, device=self.device)
        current_model_energies = self.get_energies(samples).detach()
        log_ws = torch.zeros(K, device=self.device)
        betas = torch.linspace(0, 1, n_beta, device=self.device)

        bit_to_flip = 0

        for beta in betas:
            bit_to_flip = (bit_to_flip + 1) % self.n

            old_energies = current_model_energies * beta

            samples[:, bit_to_flip] = 1 - samples[:, bit_to_flip]

            proposed_model_energies = self.get_energies(samples).detach()
            proposed_energies = proposed_model_energies * beta

            transition_ps = torch.exp(current_energies - proposed_energies)
            random_tensor = torch.rand(K, device=transition_ps.device)
            reject_transitions = (transition_ps < random_tensor).type(torch.get_default_dtype())

            samples[:, bit_to_flip] = torch.abs(samples[:, bit_to_flip] - reject_transitions)

            current_model_energies = reject_transitions * current_model_energies + (
                    1 - reject_transitions) * proposed_model_energies

            current_energies = current_model_energies * beta

            log_ws += current_energies - old_energies

        log_ws -= current_energies

        log_z = self.n * torch.tensor(2, device=self.device, dtype=torch.get_default_dtype()).log()
        log_z -= torch.tensor(K, device=self.device, dtype=torch.get_default_dtype()).log()

        # Normalizing log_ws to avoid inf if the values are too small or too large
        log_ws_min = log_ws.min()
        normalized_log_ws = log_ws - log_ws_min

        log_z += log_ws_min
        # in the future might need to change normalized_log_ws to float64 if the values are too large or too small.
        log_z += torch.log(torch.exp(normalized_log_ws).sum())

        if torch.isinf(log_z):
            warnings.warn('log_z is inf, check AIS function. log_ws values might be to high to exponentiate.')

        return log_z

    def get_log_probability(self, words: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculates the predicted log-probabilities of the model for the given words.
        :param words: Words to calculate the log probability for, if not provided will use self.all_words.
        :return: the log-probabilities of the given words
        """
        energies = self.get_energies(words)

        if self.log_z is None:
            self.calculate_z()

        log_probabilities = -energies - self.log_z

        return log_probabilities

    def get_probability(self, words: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculates the predicted probabilities of the model for the given words.
        :param words: Words to calculate the probability for, if not provided will use self.all_words.
        :return: the probabilities of the given words
        """
        log_probabilities = self.get_log_probability(words)

        return torch.exp(log_probabilities)

    def calculate_entropy(self) -> float:
        """
        Calculates the entropy of the model (exhaustive)
        :return: the entropy of the model in bits
        """
        ps = self.get_probability()
        log_ps = torch.log2(ps)

        entropy = - (ps * log_ps).sum()

        return entropy.item()

    def generate_samples(self, n_samples: int, burnin: int = 1, separation: Optional[int] = None,
                         sampler_mat_size: Optional[int] = 20000,
                         initial_word: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generating samples using Metropolis-Hastings algorithm
        https://en.wikipedia.org/wiki/Metropolis-Hastings_algorithm
        :param n_samples: number of samples to generate
        :param burnin: number of steps to perform from the initial state before saving first sample.
        Apparently burn-in is not necessarily needed, so default is set to just 1:
        http://users.stat.umn.edu/~geyer/mcmc/burn.html
        :param separation: number of steps to preform between consecutive samples. default is model.n
        :param sampler_mat_size: number of words to generate at each MCMC step.
        Larger values would run faster but require more memory, and there's a chance the the generated samples
        would be biased
        :param initial_word: the initial sampler state to begin sampling with. Default is set to all zeros.
        :return: the generated samples
        """
        if separation is None:
            separation = self.n

        if initial_word is None:
            # initial state is all zeros, ideally should start with the highest probability word, but it's usually all
            # zeros and for large n searching for highest probability word can take too long
            initial_word = torch.zeros(self.n, device=self.device)

        # initializing sampler
        if self.sampler_state is None:
            self.sampler_state = torch.stack([initial_word] * sampler_mat_size)
        elif self.sampler_state.shape[0] < sampler_mat_size:
            n_extra_rows = sampler_mat_size - self.sampler_state.shape[0]
            extra_rows = torch.stack([initial_word] * n_extra_rows)
            self.sampler_state = torch.cat([self.sampler_state, extra_rows])
        else:
            self.sampler_state = self.sampler_state[:sampler_mat_size]

        # calculating number of iterations
        iterations = int(math.ceil(n_samples / sampler_mat_size))

        samples = self.generate_samples_sequential_separation(iterations, burnin, separation)

        return samples[:n_samples]

    def generate_samples_sequential_separation(self, iterations: int, burnin: int, separation: int) -> torch.Tensor:
        """
        Generating samples using metropolis-Hastings algorithm, with sequential bit flipping between
        consecutive samples.
        https://en.wikipedia.org/wiki/Metropolis-Hastings_algorithm
        :param iterations: number of MCMC steps to perform (excluding separation)
        :param burnin: how many burnin iteration to perform (multiplied separation)
        :param separation: how many mcmc steps to execute between each saved sample
        :return: the generated samples
        """
        # setting first bit to flip
        bit_to_flip = 0
        sampler_current_energies = self.get_energies(self.sampler_state)

        # burn-in process, doing burnin x separation mcmc steps on sampler.
        sampler_current_energies, bit_to_flip = self.sampler_do_n_mcmc_steps(sampler_current_energies, bit_to_flip,
                                                                             burnin * separation)

        # sampling process.
        samples_list = []

        bit_to_flip = 0
        for _ in it.repeat(None, iterations):
            sampler_current_energies, bit_to_flip = self.sampler_do_n_mcmc_steps(sampler_current_energies, bit_to_flip,
                                                                                 separation)
            # Saving current sampler state
            samples_list.append(self.sampler_state.clone())

        # concatenating saved samples to one large tensor
        samples = torch.cat(samples_list)
        return samples

    def sampler_do_n_mcmc_steps(self, sampler_current_energies: torch.Tensor, bit_to_start: int, n_steps: int) -> Tuple[
        torch.Tensor, int]:
        """
        performs n_steps MCMC steps and then returns the updated sampler state with the updates energies.
        each MCMC step the flipped bit is incremented by 1 (modulo n)
        :param sampler_current_energies: the sampler current energies
        :param bit_to_start: the bit to
        :param n_steps: number of MCMC steps to execute
        :return: the sampler updated energies and the next bit to flip
        """
        bit_to_flip = bit_to_start
        for _ in it.repeat(None, n_steps):
            sampler_current_energies = self.sampler_next_step(bit_to_flip, sampler_current_energies)
            #  moving to next bit in word reciprocally
            bit_to_flip = (bit_to_flip + 1) % self.n
        return sampler_current_energies, bit_to_flip

    def sampler_next_step(self, bit_to_flip: int, current_energies: torch.Tensor) -> torch.Tensor:
        """
        This function does a metropolis Hastings MCMC step on sampler_state.
        https://en.wikipedia.org/wiki/Metropolis-Hastings_algorithm
        :param bit_to_flip: int between 0 and model.n. the index of the bit to flip
        :param current_energies: the energies of the current sampler state.
        :return: the new sampler_state energies after the MCMC step.
        """
        # flipping bit and getting new proposed energy
        self.sampler_state[:, bit_to_flip] = 1 - self.sampler_state[:, bit_to_flip]
        proposed_energies = self.get_energies(self.sampler_state).detach()

        # getting transition probability and getting rejections
        transition_ps = torch.exp(current_energies - proposed_energies)
        sampler_mat_size = self.sampler_state.shape[0]
        random_tensor = torch.rand(sampler_mat_size, device=transition_ps.device)
        reject_transitions = (transition_ps < random_tensor).type(torch.get_default_dtype())

        # switching back if should reject
        self.sampler_state[:, bit_to_flip] = torch.abs(self.sampler_state[:, bit_to_flip] - reject_transitions)

        # Calculating and returning energies of current sampler state
        current_energies = reject_transitions * current_energies + (1 - reject_transitions) * proposed_energies
        return current_energies.detach()

    def get_empirical_std(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Calculating the empirical standard deviation of the given samples using a Clopper-Pearson confidence interval
        of one sigma (0.32).
        Clopper-Pearson is a conservative estimate, alternatively we can simply compute the sample standard deviation.
        :param samples: the empirical samples
        :return: the empirical samples one standard deviation confidence interval
        """
        n_samples = samples.shape[0]
        empirical_marginals = self.get_empirical_marginals(samples)
        return self.clopper_pearson_confidence_interval(empirical_marginals, n_samples)

    def clopper_pearson_confidence_interval(self, empirical_marginals: torch.Tensor, n_samples: int,
                                            alpha: float = 0.32) -> torch.Tensor:
        """
        http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
        alpha confidence intervals for a binomial distribution of k expected successes on n trials
        Clopper Pearson intervals are a conservative estimate.
        """
        # moving empirical marginal to cpu (if its already there nothing happens) for the
        # clopper pearson function
        lower, upper = self.clopper_pearson(empirical_marginals.cpu() * n_samples, n_samples, alpha)

        # moving the result back to the device of the empirical marginals
        limits_matrix = torch.cat((lower[None, :], upper[None, :]), 0).to(empirical_marginals.device)
        intervals_matrix = torch.abs(limits_matrix - empirical_marginals)
        sigmas = intervals_matrix.max(0)[0]

        return sigmas

    def clopper_pearson(self, k: torch.Tensor, n: int, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
        alpha confidence intervals for a binomial distribution of k expected successes on n trials
        Clopper Pearson intervals are a conservative estimate.
        """
        lo = scipy.stats.beta.ppf(alpha / 2, k, n - k + 1)
        lo[np.isnan(lo)] = 0  # hack to remove NaNs where we only have 0 samples
        hi = scipy.stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
        hi[np.isnan(hi)] = 1  # hack to remove NaNs where the marginal is 1
        return torch.from_numpy(lo), torch.from_numpy(hi)

    def move_samples_to_device(self, samples: ArrayOrTensor) -> torch.Tensor:
        """
        Converts samples from np array to torch tensor, and moves it to model.device
        :param samples: samples to convert and move
        :return: samples as torch tensor on model.device
        """
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples).type(torch.get_default_dtype()).to(self.device)
        elif not isinstance(samples, torch.Tensor):
            raise TypeError('type(samples) must be torch.Tensor or a numpy.ndarray')

        if samples.shape[1] != self.n:
            raise ValueError(f'Second dimension in samples must be equal to number of cells. {samples.shape[1]} '
                             f'provided, expected {self.n}')

        return samples.to(self.device)

    # Abstract methods to be implemented by subclasses
    @abstractmethod
    def get_marginals(self) -> torch.Tensor:
        """
        Abstract method that should be implemented by each model subclass.
        Calculates the marginals of the model (exhaustive).
        :return: model predicted marginals
        """
        pass

    @abstractmethod
    def get_energies(self, words: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Abstract method that should be implemented by each model subclass.
        Calculates the energies vector of the model for the given words, if words is none than computes the energies
        on all possible words
        :param words: words to calculate energy on
        :return: energies of the given words
        """
        pass

    @abstractmethod
    def get_empirical_marginals(self, samples: ArrayOrTensor) -> torch.Tensor:
        """
        Abstract method that should be implemented by each model subclass.
        Calculates and returns the empirical marginals of the model from the provided samples
        :param samples: empirical samples to calculate marginals on
        :return: the empirical marginals of the given samples
        """
        pass
