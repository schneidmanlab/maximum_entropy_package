# maximum_entropy_package
`maximum_entropy_package` is a Python package designed to create maximum entropy models for neural population codes based on empirical data. The package allows you to construct independent, pairwise, and random projection models, leveraging GPU acceleration through PyTorch for efficient computations.

Maximum entropy models are statistical models that make the least biased predictions without assuming any additional structure. In the context of neural population codes, these models provide a principled way to infer the distribution of neural activity patterns that is most consistent with the observed data. 

Mathematically, given a set of binary activity patterns $\\{ \vec{x} \\}$ (where `0` indicates a silent neuron and `1` indicates an active one) the maximum entropy model $\hat{p}$ which is consistent with the average values of a set of functions $f_1 \ldots f_K$ is given by:

$$
\hat{p}(\vec{x}) = 
\frac{1}{Z} \exp \left(- \sum_{i=k}^{K} \lambda_k f_k (\vec{x}) \right),
$$

where $\lambda_1 \ldots \lambda_k$ are the model parameters which can be found numerically. 

The package provides several Maximum Entropy models:
- **Independent Model**: Construct maximum entropy models based on the observed firing rates of individual neurons $\left< x_i \right>$.
- **[Pairwise Model](https://www.nature.com/articles/nature04701)**: Construct maximum entropy models based on firing rates $\left< x_i \right>$ and pairwise co-firing rates $\left< x_i x_j\right>$.
- **[Random Projections](https://www.pnas.org/doi/10.1073/pnas.1912804117)**: Construct maximum entropy models using sparse non-linear projections over the data $\left< \Theta(a_{ij} x_j - \theta_i) \right>$, where $\Theta$ denotes a step function non-linearity.
- **[Sigmoid Random Projections](https://elifesciences.org/reviewed-preprints/96566)**: Same as the random projections models, with a sigmoid non-linearity $\left< \sigma(a_{ij} x_j - \theta_i) \right>$. 
  
## Usage example
#### Learning maximum entropy models for small populations ($n<25$)
```
import maximumentropy as me
import torch
import numpy as np

# load spikes
spikes = torch.from_numpy(np.load('example_raster_20.npz')['spikes']).type(torch.get_default_dtype())

n = spikes.shape[1]
nsamples = spikes.shape[0]

# create a maximum entropy model 
model = me.Pairwise(n)

# or random projections model with default settings:
# model = me.RP(n)
# model = me.SigmoidMERP(n)

# get empirical marginals <f_i>_emp
emp_marginals = model.get_empirical_marginals(spikes)

# train the model, for n<25 model is trained exhaustively, probability of all possible states is stored in memory
model.train(spikes)

# for small population we can calculates the model predicted marginals
predicted_marginals = model.get_marginals()

# get model predicted log probabilities
predicted_log_probs = model.get_log_probability(spikes)
```


#### Learning maximum entropy models for large populations ($n>25$)

```
import maximumentropy as me
import torch
import numpy as np

# load spikes
spikes = torch.from_numpy(np.load('example_raster_50.npz')['spikes']).type(torch.get_default_dtype())

n = spikes.shape[1]
nsamples = spikes.shape[0]

# create a maximum entropy model 
model = me.Pairwise(n)

# or random projections model with default settings:
# model = me.RP(n)
# model = me.SigmoidMERP(n)

# get empirical marginals <f_i>_emp
emp_marginals = model.get_empirical_marginals(spikes)

# train the model, for n>25 model is trained using Markov chain Monte Carlo methods (MCMC)
model.train(spikes)

# for lareg populations we cannot calculates the model predicted marginals, 
# instead we can evaluate the marginals by sampling from the model and calculating the sampled data marginals
mcmc_samples = model.generate_samples(nsamples)
mcmc_marginals = model.get_empirical_marginals(mcmc_samples)

# calculating probabilities requires an approximation of the partition function Z
# which is done using annealed importance sampling, for large models (n>200) the default
# params of AIS might not provide an accurate estimate and larger values would 
# be required. e.g.:
model.calculate_z(n_beta=50_000)

# get model predicted log probabilities (using the approximated log_z)
predicted_log_probs = model.get_log_probability(spikes)
```

#### Loading an existing model

Loading a pairwise model with existing factors:
```
# loading existing factors
factors = ...

# create a maximum entropy pairwise model with predetermined factors
model = me.Pairwise(n, factors=factors)
```

Loading RP and Sigmoid RP models with existing projections:
```
# loading projections and thresholds
projections = ...
projections_thresholds = ...

# existing factors can also be provided
factors = ...

model = me.RP(n, projections=projections, 
              projections_thresholds=projections_thresholds, 
              factors=factors)
```

## cite 
If you use `maximum_entropy_package` in your research, please cite it as follows:

```
@software{maximum_entropy_package,
  author = {Jonathan Mayzel},
  title = {maximum_entropy_package},
  year = {2024},
  url = {https://github.com/yonimayzel/maximum_entropy_package},
}
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Authors and Contributors
Jonathan Mayzel, yoni.mayzel@gmail.com