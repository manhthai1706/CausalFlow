"""
CausalFlow: Advanced Multivariate Causal Discovery Framework
"""

__version__ = '1.0.0'

# Export main classes to top-level
from .models.causalflow import CausalFlow
from .models.analysis import CausalAnalyzer, ANMMM_cd_advanced, ANMMM_clu
from .models.trainer import CausalFlowTrainer

# Export core components for advanced users
from .core.gppom_hsic import GPPOMC_lnhsic_Core, FastHSIC
from .core.hsic import hsic_gam

__all__ = [
    'CausalFlow',
    'CausalAnalyzer',
    'ANMMM_cd_advanced',
    'ANMMM_clu',
    'CausalFlowTrainer',
    'GPPOMC_lnhsic_Core',
    'FastHSIC',
    'hsic_gam'
]
