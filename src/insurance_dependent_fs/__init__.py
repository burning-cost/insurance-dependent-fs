"""
insurance-dependent-fs
======================
Dependent frequency-severity neural two-part model for insurance pricing.

The core idea is multi-task learning: a single shared encoder trunk processes
covariates and produces a latent representation that feeds both a Poisson
frequency head and a Gamma severity head.  Gradients from both losses flow
through the trunk simultaneously, so it learns features that are jointly
informative for frequency *and* severity.  That shared information is where the
implicit frequency-severity dependence lives.

On top of the latent dependence you can add the explicit Garrido-Genest-Schulz
conditional covariate (log μ += γ·N), which gives a semi-analytical pure
premium correction and a directly interpretable dependence parameter.

The library is distinct from ``insurance-frequency-severity`` (Sarmanov copula,
parametric, scipy-based).  Use Sarmanov when you need an analytical bivariate
density and a simple omega parameter for a regulator; use this library when you
have a large dataset, suspect nonlinear interactions, and want a single model
that learns both tasks jointly.

Public API
----------
Model
~~~~~
``DependentFreqSevNet``   – PyTorch nn.Module (shared trunk + heads)
``SharedTrunkConfig``     – dataclass for trunk hyperparameters
``FrequencyHead``         – Poisson head module
``SeverityHead``          – Gamma head module

Training
~~~~~~~~
``JointLoss``             – Poisson + Gamma NLL with configurable balancing
``DependentFSTrainer``    – training loop with early stopping and LR scheduling

Wrapper
~~~~~~~
``DependentFSModel``      – sklearn-compatible estimator (fit/predict/score)

Premium
~~~~~~~
``PurePremiumEstimator``  – Monte Carlo + optional MGF analytical correction

Diagnostics
~~~~~~~~~~~
``DependentFSDiagnostics`` – Lorenz, calibration, dependence tests, latent corr

Data
~~~~
``FreqSevDataset``        – PyTorch Dataset with exposure handling
``prepare_features``      – numeric encoding helper

Benchmarks
~~~~~~~~~~
``make_dependent_claims`` – synthetic claims with known γ dependence
``make_independent_claims`` – synthetic independent baseline
"""

from insurance_dependent_fs.model import (
    DependentFreqSevNet,
    FrequencyHead,
    SeverityHead,
    SharedTrunkConfig,
)
from insurance_dependent_fs.training import DependentFSTrainer, JointLoss
from insurance_dependent_fs.wrapper import DependentFSModel
from insurance_dependent_fs.premium import PurePremiumEstimator
from insurance_dependent_fs.diagnostics import DependentFSDiagnostics
from insurance_dependent_fs.data import FreqSevDataset, prepare_features
from insurance_dependent_fs.benchmarks import make_dependent_claims, make_independent_claims

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("insurance-dependent-fs")
except PackageNotFoundError:
    __version__ = "0.0.0"  # not installed

__all__ = [
    "DependentFreqSevNet",
    "FrequencyHead",
    "SeverityHead",
    "SharedTrunkConfig",
    "JointLoss",
    "DependentFSTrainer",
    "DependentFSModel",
    "PurePremiumEstimator",
    "DependentFSDiagnostics",
    "FreqSevDataset",
    "prepare_features",
    "make_dependent_claims",
    "make_independent_claims",
]
