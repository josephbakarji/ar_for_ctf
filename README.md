# Polynomial Autoregressive (AR) Model

This directory contains an implementation of *Polynomial Autoregressive* modeling for [CTF-for-Science](https://github.com/CTF-for-Science).

## Model Description

### Basic Autoregressive Models
Standard linear autoregressive (AR) models predict future values based on past observed values. In a linear AR model of order p, denoted as AR(p), the current value is expressed as a linear combination of p previous values plus a random error term:

$$y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + ... + \phi_p y_{t-p} + \varepsilon_t$$

where:
- $y_t$ is the value at time t
- $c$ is a constant
- $\phi_1, \phi_2, ..., \phi_p$ are the parameters of the model
- $\varepsilon_t$ is white noise

### Polynomial Autoregressive Models
This implementation extends the standard AR model to include polynomial terms. A polynomial AR model incorporates nonlinear relationships by including higher-order terms and cross-products of lagged values:

$$y_t = c + \sum_{i=1}^{p} \phi_i y_{t-i} + \sum_{i=1}^{p}\sum_{j=i}^{p} \phi_{ij} y_{t-i}y_{t-j} + \sum_{i=1}^{p}\sum_{j=i}^{p}\sum_{k=j}^{p} \phi_{ijk} y_{t-i}y_{t-j}y_{t-k} + ... + \varepsilon_t$$

For example, a polynomial AR model of order 2 (lag=2) with polynomial degree 2 would include:
- Linear terms: $y_{t-1}$, $y_{t-2}$
- Quadratic terms: $y_{t-1}^2$, $y_{t-2}^2$
- Cross-terms: $y_{t-1}y_{t-2}$

These cross-terms and higher-order polynomial terms capture complex nonlinear dynamics that simple linear AR models cannot represent. This makes polynomial AR models particularly powerful for modeling complex dynamical systems with nonlinear behavior.

When applied to spatial-temporal dynamical systems, AR models are fitted to each spatial point independently, allowing for the prediction of complex behavior while maintaining computational efficiency.

For higher-dimensional systems, this implementation also supports dimensionality reduction using Proper Orthogonal Decomposition (POD) to reduce computational cost while preserving important dynamics.

This implementation includes support for both standard AR and ARIMA (Autoregressive Integrated Moving Average) models, with special handling for various prediction scenarios including reconstruction tasks and potentially unstable systems.

## Files
- `AR.py`: Contains the `AR` class implementing the polynomial autoregressive model logic using [statsmodels](https://www.statsmodels.org/).
- `run.py`: Batch runner script for running the model across multiple sub-datasets in the [CTF-for-Science](https://github.com/CTF-for-Science) framework.
- `run_opt.py`: Batch runner script for running the model across multiple sub-datasets with hyperparameter tuning.
- `optimize_parameters.py`: Script for tuning the model hyperparameters.
- `config/config_KS.yaml`: Configuration file for running the model on the `Kuramoto-Sivashinsky` test cases.
- `config/config_Lorenz.yaml`: Configuration file for running the model on the `Lorenz` test cases.
- `tuning_config/config_KS.yaml`: Configuration file for tuning the model hyperparameters on the `Kuramoto-Sivashinsky` test cases.
- `tuning_config/config_Lorenz.yaml`: Configuration file for tuning the model hyperparameters on the `Lorenz` test cases.
 
The configuration files in the `config` folder specify the hyperparameters for running the model with the following structure:

```yaml
dataset:
  name: <dataset_name>  # Test case (e.g. PDE_KS, ODE_Lorenz)
  pair_id: 'all'        # Which sub-datasets to consider
model:
  name: AR
  lag: <lag>                    # Number of past timesteps to consider (p)
  poly_degree: <poly_degree>    # Degree of polynomial terms (maximum exponent in the model)
  trend: <trend>                # Trend component ('n', 'c', 't', 'ct')
  seasonal: <seasonal>          # Boolean for seasonal component
  POD_modes: <number_POD_modes> # Number of POD modes for dimensionality reduction
  use_arima: <use_arima>        # Boolean to use ARIMA instead of AR
  d: <d>                        # Differencing order for ARIMA
  q: <q>                        # Moving average order for ARIMA
```

The configuration files in the `tuning_config` folder specify the possible hyperparameter values to explore while tuning them. 

## Usage

In the [CTF-for-Science](https://github.com/CTF-for-Science) framework, the AR model can be tested with the command:

```bash
python models/ar_model/run.py models/ar_model/config/config_*.yaml
```

For hyperparameter tuning:

```bash
python models/ar_model/optimize_parameters.py models/ar_model/tuning_config/config_*.yaml
```

## Special Cases Handling

This implementation includes special handling for different types of tasks:

1. **Reconstruction Tasks (Pair IDs 2, 4)**: For these tasks, the lag parameter is automatically set to 0, as burn-in data is not needed.

2. **Unstable Prediction (Pair ID 6)**: For potentially unstable predictions, the model uses chunked forecasting, value clipping, and smoothing techniques to stabilize results.

3. **High-Dimensional Data**: For datasets with many spatial points (like PDE_KS), POD is used to reduce dimensions while preserving dynamics.

## Dependencies
- numpy
- scikit-learn
- statsmodels
- pyyaml
- matplotlib

