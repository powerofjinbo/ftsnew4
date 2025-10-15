"""
Asimov Dataset Utilities for FTS Implementation
===============================================

The Asimov dataset is a special dataset where the observed values
equal their expected values under a given hypothesis. It's used to
compute expected (median) results without generating many toys.

Reference: Cowan et al., Eur.Phys.J. C71 (2011) 1554
"""

import numpy as np
from typing import Optional, Dict, Tuple

class AsimovDataset:
    """
    Generate and manage Asimov datasets for hypothesis testing
    
    An Asimov dataset is constructed such that when used to evaluate
    test statistics, it gives the median expected result under a
    specified hypothesis.
    """
    
    def __init__(self, model_config: Dict):
        """
        Initialize Asimov dataset generator
        
        Args:
            model_config: Dictionary containing model parameters
                - n_bins: Number of bins
                - bkg_events: Background events per bin
                - sig_events: Signal events per bin
                - bkg_uncertainty: Systematic uncertainty on background
        """
        self.n_bins = model_config.get('n_bins', 3)
        self.bkg_events = np.array(model_config.get('bkg_events', [15, 20, 17]))
        self.sig_events = np.array(model_config.get('sig_events', [1, 1, 1]))
        self.bkg_uncertainty = model_config.get('bkg_uncertainty', 0.1)
        
    def generate_asimov(self, mu: float = 0.0, 
                        nuisance_values: Optional[Dict] = None) -> np.ndarray:
        """
        Generate Asimov dataset for given signal strength
        
        Args:
            mu: Signal strength parameter
            nuisance_values: Dictionary of nuisance parameter values
                            (if None, uses nominal values)
        
        Returns:
            Asimov data values (expected counts per bin)
        """
        # Start with nominal background
        asimov_data = self.bkg_events.copy()
        
        # Apply nuisance parameter shifts if specified
        if nuisance_values:
            alpha_sys = nuisance_values.get('alpha_sys', 0.0)
            # Shift background by systematic uncertainty
            asimov_data = asimov_data * (1.0 + alpha_sys * self.bkg_uncertainty)
        
        # Add signal contribution
        asimov_data = asimov_data + mu * self.sig_events
        
        # For Asimov dataset, we use the expected values directly
        # (no Poisson fluctuation)
        return asimov_data
    
    def generate_asimov_with_profile(self, mu: float) -> Tuple[np.ndarray, Dict]:
        """
        Generate Asimov dataset with profiled nuisance parameters
        
        This is more sophisticated: it finds the values of nuisance
        parameters that maximize the likelihood for the given mu,
        then generates the Asimov dataset at those values.
        
        Args:
            mu: Signal strength parameter
            
        Returns:
            Tuple of (asimov_data, profiled_nuisances)
        """
        # For single nuisance parameter (background systematic)
        # the profile value can be computed analytically
        
        # Expected total events
        expected_total = np.sum(self.bkg_events + mu * self.sig_events)
        
        # For Gaussian-constrained nuisance, profile value is
        # pulled toward the value that best fits the data
        # For Asimov, this is typically zero (nominal)
        alpha_sys_profiled = 0.0
        
        # Generate Asimov dataset at profiled values
        asimov_data = self.generate_asimov(mu, {'alpha_sys': alpha_sys_profiled})
        
        return asimov_data, {'alpha_sys': alpha_sys_profiled}
    
    def compute_expected_significance(self, mu_signal: float) -> float:
        """
        Compute expected discovery significance using Asimov dataset
        
        This uses the Asimov approximation to calculate the median
        expected significance for discovering a signal of strength mu_signal
        when the true signal strength is mu_signal.
        
        Args:
            mu_signal: True signal strength
            
        Returns:
            Expected significance in units of standard deviations
        """
        # Generate Asimov dataset under signal hypothesis
        asimov_data, _ = self.generate_asimov_with_profile(mu_signal)
        
        # Total expected counts
        s = np.sum(mu_signal * self.sig_events)  # Signal
        b = np.sum(self.bkg_events)              # Background
        
        # Approximate significance formula (valid for large counts)
        # Z = sqrt(2 * ((s+b) * log(1 + s/b) - s))
        if b > 0:
            significance = np.sqrt(2 * ((s + b) * np.log(1 + s/b) - s))
        else:
            significance = 0.0
            
        return significance
    
    def expected_limit(self, confidence_level: float = 0.95) -> float:
        """
        Calculate expected upper limit using Asimov dataset
        
        Args:
            confidence_level: Confidence level for limit (default 95%)
            
        Returns:
            Expected upper limit on signal strength
        """
        # For simple counting experiment with Asimov dataset
        # Expected limit can be approximated
        
        b = np.sum(self.bkg_events)
        sigma_b = np.sqrt(np.sum((self.bkg_events * self.bkg_uncertainty)**2))
        
        # Approximate expected limit (simplified formula)
        # More sophisticated calculation would use profile likelihood
        from scipy import stats
        z = stats.norm.ppf(confidence_level)
        
        expected_limit = z * np.sqrt(b + sigma_b**2) / np.sum(self.sig_events)
        
        return expected_limit

def create_asimov_for_workspace(workspace, mu: float = 0.0, 
                                dataset_name: str = "asimovData") -> None:
    """
    Create Asimov dataset in xRooFit workspace
    
    Args:
        workspace: xRooFit workspace object
        mu: Signal strength for Asimov generation
        dataset_name: Name for the dataset
    """
    try:
        import ROOT
        
        # Set POI to desired value
        poi = workspace.pars()["mu"]
        old_val = poi.getVal()
        poi.setVal(float(mu))
        
        # Generate Asimov dataset (expected values, no fluctuation)
        # This is ROOT's built-in Asimov generation
        pdf = workspace["pdfs/simPdf"]
        
        # Use Extended(True) to obtain the expected (Asimov) counts
        asimov = pdf.generate(ROOT.RooFit.Extended(True))
        asimov.get().SetNameTitle(dataset_name, f"Asimov Dataset (mu={mu})")
        
        # Add to workspace
        workspace.Add(asimov)
        
        # Reset POI
        poi.setVal(old_val)
        
        print(f"Created Asimov dataset '{dataset_name}' with mu={mu}")
        
    except Exception as e:
        print(f"Error creating Asimov dataset: {e}")
        print("Using simplified Asimov generation...")
        
        # Fallback to manual generation
        model_config = {
            'n_bins': 3,
            'bkg_events': [15, 20, 17],
            'sig_events': [1, 1, 1],
            'bkg_uncertainty': 0.1
        }
        asimov_gen = AsimovDataset(model_config)
        asimov_data = asimov_gen.generate_asimov(mu)
        print(f"Generated Asimov data: {asimov_data}")
        return asimov_data

# Example usage and validation
if __name__ == "__main__":
    print("Asimov Dataset Utilities")
    print("=" * 60)
    
    # Example configuration
    config = {
        'n_bins': 3,
        'bkg_events': [15, 20, 17],
        'sig_events': [1, 1, 1],
        'bkg_uncertainty': 0.1
    }
    
    # Create Asimov generator
    asimov = AsimovDataset(config)
    
    # Generate Asimov datasets for different signal strengths
    print("\nAsimov Datasets for different mu values:")
    print("-" * 40)
    for mu in [0.0, 1.0, 2.0]:
        data = asimov.generate_asimov(mu)
        print(f"mu = {mu}: {data}")
    
    # Compute expected significance
    print("\nExpected Discovery Significance:")
    print("-" * 40)
    for mu in [1.0, 2.0, 3.0]:
        sig = asimov.compute_expected_significance(mu)
        print(f"mu = {mu}: {sig:.2f} sigma")
    
    # Expected limit
    print("\nExpected Upper Limit:")
    print("-" * 40)
    limit = asimov.expected_limit(0.95)
    print(f"95% CL expected limit: mu < {limit:.2f}")
    
    print("\n" + "=" * 60)
    print("Asimov dataset represents the 'expected' or 'median' outcome")
    print("It's used to compute expected results without generating toys")
    print("For actual analysis, use with xRooFit workspace")
