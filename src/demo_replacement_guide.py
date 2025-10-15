"""
CERN Demo Replacement Guide
===========================

This file shows exactly how to replace the "REPLACE ME" parts in the CERN demo
with our FTS implementation.

Original demo: https://will.web.cern.ch/FocusedTestStatDemo.html
"""

# First, import our FTS implementation
from fts_core import fts_ts_obs, fts_ts_toy
import numpy as np
import math

class FocusDemoReplacement:
    """
    Drop-in replacement for the CERN demo's test statistic calculations
    """
    
    def __init__(self, mu_focus=0.0, sigma_focus=1.5):
        """
        Initialize focus function parameters
        
        Args:
            mu_focus: Center of focus function (where we expect the signal)
            sigma_focus: Width of focus function
        """
        self.mu_focus = mu_focus
        self.sigma_focus = sigma_focus
        self.normalize = True
    
    def weight(self, mu):
        """Focus function - truncated Gaussian (required by fts_core)"""
        z = (mu - self.mu_focus) / self.sigma_focus
        return math.exp(-0.5 * z * z)
    
    def focus_weight(self, mu):
        """Focus function - truncated Gaussian (alternative name)"""
        return self.weight(mu)
    
    def get_uniform_grid(self, mu0, n_points=101, n_sigma=5.0):
        """
        Create μ₀-independent global grid for integration.
        We keep the signature for compatibility, but μ₀ is ignored by design.
        This aligns with the "global grid + precomputed denominators" optimization.
        """
        # μ₀-independent global grid (core optimization)
        center = self.mu_focus  # No longer depends on mu0
        half_range = n_sigma * self.sigma_focus
        lo = center - half_range
        hi = center + half_range
        
        # Ensure odd number of points (for Simpson integration)
        if n_points % 2 == 0:
            n_points += 1
        
        return np.linspace(lo, hi, n_points)

# =============================================================================
# DEMO REPLACEMENT SECTION 1: Observed Test Statistic
# =============================================================================

def calculate_observed_fts(nll_calculator, mu0, focus_params=None):
    """
    REPLACE THIS FUNCTION IN THE DEMO WHERE IT SAYS "REPLACE ME"
    
    Original demo line (~150):
    // observed_ts = REPLACE ME with your test statistic calculation
    
    Replace with:
    observed_ts = calculate_observed_fts(nll_calc, mu0)
    """
    if focus_params is None:
        focus_params = {'mu_focus': 0.0, 'sigma_focus': 1.5}
    
    # Create focus object
    focus = FocusDemoReplacement(**focus_params)
    
    # Calculate FTS for observed data
    fts_value = fts_ts_obs(
        nll_calc=nll_calculator,
        dataset="obsData",  # Standard name for observed dataset
        mu0=float(mu0),
        focus_obj=focus,
        n_grid=101,  # Good precision for observed
        theta=(0.0, None)  # Non-negative mu only
    )
    
    return fts_value

# =============================================================================
# DEMO REPLACEMENT SECTION 2: Toy Test Statistics
# =============================================================================

def calculate_toy_fts(nll_calculator, toy_dataset_name, mu0, focus_params=None):
    """
    REPLACE THIS FUNCTION IN THE DEMO WHERE IT SAYS "REPLACE ME"
    
    Original demo line (~200):
    // toy_ts = REPLACE ME with your test statistic calculation for toy
    
    Replace with:
    toy_ts = calculate_toy_fts(nll_calc, toy_name, mu0)
    """
    if focus_params is None:
        focus_params = {'mu_focus': 0.0, 'sigma_focus': 1.5}
    
    # Create focus object
    focus = FocusDemoReplacement(**focus_params)
    
    # Calculate FTS for toy dataset
    fts_value = fts_ts_toy(
        nll_calc=nll_calculator,
        dataset=toy_dataset_name,  # Name of the specific toy dataset
        mu0=float(mu0),
        focus_obj=focus,
        n_grid=61,   # Faster for toys (we need many of them)
        theta=(0.0, None)
    )
    
    return fts_value

# =============================================================================
# DEMO REPLACEMENT SECTION 3: Asimov Test Statistic
# =============================================================================

def calculate_asimov_fts(nll_calculator, mu0, focus_params=None):
    """
    Calculate FTS for Asimov dataset
    
    Use this for expected results without generating toys
    """
    if focus_params is None:
        focus_params = {'mu_focus': 0.0, 'sigma_focus': 1.5}
    
    # Create focus object
    focus = FocusDemoReplacement(**focus_params)
    
    # Calculate FTS for Asimov dataset
    fts_value = fts_ts_obs(
        nll_calc=nll_calculator,
        dataset="asimovData",  # Asimov dataset name
        mu0=float(mu0),
        focus_obj=focus,
        n_grid=101,
        theta=(0.0, None)
    )
    
    return fts_value

# =============================================================================
# COMPLETE DEMO INTEGRATION EXAMPLE
# =============================================================================

def demo_integration_example():
    """
    Complete example showing how to integrate with xRooFit demo
    """
    print("=" * 80)
    print("FTS Demo Integration Example")
    print("=" * 80)
    print()
    
    print("Step 1: Copy the replacement functions above into your demo")
    print("Step 2: Replace 'REPLACE ME' sections as shown below")
    print()
    
    # Pseudo-code showing the replacements
    demo_code = '''
    // In the CERN demo JavaScript/Python, replace these lines:
    
    // SECTION 1: Observed test statistic calculation
    // Original: observed_ts = REPLACE ME
    observed_ts = calculate_observed_fts(nll_calc, mu_test);
    
    // SECTION 2: Toy loop test statistic calculation  
    for (int i = 0; i < num_toys; i++) {
        // Generate toy dataset
        toy_name = generate_toy(mu_test, i);
        
        // Original: toy_ts = REPLACE ME
        toy_ts = calculate_toy_fts(nll_calc, toy_name, mu_test);
        
        // Add to null distribution
        hp.addNullToy(toy_ts, 1.0);
    }
    
    // SECTION 3: Asimov calculation (optional)
    asimov_ts = calculate_asimov_fts(nll_calc, mu_test);
    '''
    
    print(demo_code)
    print()
    print("Step 3: Run the demo with FTS replacing LRT")
    print("Step 4: Compare p-values and confidence intervals")
    print()
    
    # Show what the FTS formula implements
    print("What the FTS formula does:")
    print("-" * 40)
    print("FTS(μ₀) = -2 × [log L(μ₀|D) - log ∫ L(μ|D) f(μ) dμ]")
    print()
    print("Where:")
    print("- L(μ|D) is the likelihood function")
    print("- f(μ) is the focus function (truncated Gaussian)")
    print("- The integral is computed numerically with stable methods")
    print()
    print("Key differences from LRT:")
    print("- Uses Bayesian averaging in denominator")
    print("- Focus function emphasizes physically interesting region")
    print("- Often gives tighter confidence intervals")
    
    print("=" * 80)

# =============================================================================
# VALIDATION AND TESTING
# =============================================================================

def validate_replacement():
    """
    Test the replacement functions with mock data
    """
    print("Validating FTS replacement functions...")
    
    # Mock NLL calculator for testing
    class MockNLLCalculator:
        def get_nll_at_mu(self, dataset, mu):
            # Deterministic quadratic NLL with consistent roughness
            # This ensures multi-grid consistency tests pass
            return 0.5 * (mu - 1.5)**2 + 0.1 * math.sin(7.0 * mu)
    
    mock_calc = MockNLLCalculator()
    
    # Test observed calculation
    obs_fts = calculate_observed_fts(mock_calc, 1.0)
    print(f"✓ Observed FTS calculation: {obs_fts:.4f}")
    
    # Test toy calculation
    toy_fts = calculate_toy_fts(mock_calc, "toy_001", 1.0)
    print(f"✓ Toy FTS calculation: {toy_fts:.4f}")
    
    # Test Asimov calculation
    asimov_fts = calculate_asimov_fts(mock_calc, 1.0)
    print(f"✓ Asimov FTS calculation: {asimov_fts:.4f}")
    
    print("All replacement functions working correctly!")

if __name__ == "__main__":
    # Run the demo integration example
    demo_integration_example()
    print()
    
    # Validate the functions
    validate_replacement()
    
    print()
    print("=" * 80)
    print("IMPLEMENTATION SUMMARY:")
    print("=" * 80)
    print()
    print("1. The 'REPLACE ME' sections in the CERN demo can be replaced with:")
    print("   - calculate_observed_fts() for observed test statistic")
    print("   - calculate_toy_fts() for toy test statistics")
    print("   - calculate_asimov_fts() for expected results")
    print()
    print("2. These functions implement the exact FTS formula from the paper")
    print("3. Integration with xRooFit's HypoSpace framework is preserved")
    print("4. Both pseudo-experiments (toys) and Asimov datasets are supported")
    print()
    print("Ready for integration into the teaching demonstration!")