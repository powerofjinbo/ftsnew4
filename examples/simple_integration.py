#!/usr/bin/env python3
"""
Simple FTS Integration Example
==============================

Demonstrates basic usage of FTS (Focused Test Statistics) implementation
Provides minimal setup example for researchers wanting to integrate FTS in their analyses.

Usage: python examples/simple_integration.py
"""

import sys
import os
import math
import numpy as np

# Add src to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

def simple_fts_example():
    """
    Simple example showing how to use FTS without xRooFit
    Uses mock data for demonstration
    """
    print("=" * 60)
    print("Simple FTS Integration Example")
    print("=" * 60)
    
    # Mock focus function for demonstration
    class MockFocusFunction:
        def __init__(self, mu_focus=1.0, sigma_focus=1.5):
            self.mu_focus = mu_focus
            self.sigma_focus = sigma_focus
            self.normalize = True
            
        def weight(self, mu):
            """Gaussian focus function"""
            z = (mu - self.mu_focus) / self.sigma_focus
            return math.exp(-0.5 * z * z)
        
        def get_uniform_grid(self, mu0, n_points=101):
            """Generate integration grid around mu0 and focus center"""
            center = (mu0 + self.mu_focus) / 2.0
            width = 3.0 * self.sigma_focus + abs(mu0 - self.mu_focus)
            lo = center - width
            hi = center + width
            return np.linspace(lo, hi, n_points)
    
    # Mock NLL calculator
    class MockNLLCalculator:
        def __init__(self, true_mu=1.2, noise_level=0.05):
            self.true_mu = true_mu
            self.noise_level = noise_level
            
        def get_nll_at_mu(self, dataset, mu):
            """Mock NLL: quadratic function around true value with noise"""
            base_nll = 0.5 * (mu - self.true_mu)**2
            # Add small random fluctuations
            noise = self.noise_level * np.random.normal()
            return base_nll + noise
    
    # Import FTS core functions
    try:
        import fts_core
        print("✅ FTS core module imported successfully")
        
        # Verify core functions
        print("\n🔍 Validating FTS core functions...")
        has_fts_ts_obs = hasattr(fts_core, 'fts_ts_obs')
        print(f"   fts_ts_obs available: {has_fts_ts_obs}")
        
    except ImportError as e:
        print(f"❌ Failed to import FTS core: {e}")
        print("Make sure you're running from the project root directory")
        return
    
    # Setup example parameters
    print("\n📊 Setting up example analysis...")
    
    # Create focus function (interested in mu around 1.0)
    focus = MockFocusFunction(mu_focus=1.0, sigma_focus=1.5)
    print(f"Focus: μ_center={focus.mu_focus}, σ={focus.sigma_focus}")
    
    # Create mock data scenario
    nll_calc = MockNLLCalculator(true_mu=1.2, noise_level=0.02)
    print(f"Mock data: true_μ={nll_calc.true_mu}, noise_level={nll_calc.noise_level}")
    
    # Test different hypothesis values
    test_hypotheses = np.arange(0.0, 3.0, 0.5)
    print(f"\n🧪 Testing hypotheses: {test_hypotheses}")
    
    print(f"\n{'Hypothesis μ₀':>15} | {'FTS Value':>12} | {'Status':>10}")
    print("-" * 45)
    
    results = []
    for mu0 in test_hypotheses:
        try:
            # Calculate FTS test statistic
            fts_value = fts_core.fts_ts_obs(
                nll_calc=nll_calc,
                dataset="mock_data",
                mu0=mu0,
                focus_obj=focus,
                n_grid=51,  # Smaller grid for speed
                theta=(-5.0, 5.0)  # Parameter bounds
            )
            
            results.append((mu0, fts_value))
            status = "✅ OK" if fts_value < 10 else "⚠ High"
            print(f"{mu0:>15.1f} | {fts_value:>12.3f} | {status:>10}")
            
        except Exception as e:
            print(f"{mu0:>15.1f} | {'Failed':>12} | {'❌ Error':>10}")
    
    # Results analysis
    if results:
        print(f"\n📈 Analysis Summary:")
        print(f"   • Computed FTS for {len(results)} hypotheses")
        
        # Find minimum FTS (best fit region)
        min_fts_mu, min_fts_val = min(results, key=lambda x: x[1])
        print(f"   • Minimum FTS: {min_fts_val:.3f} at μ₀={min_fts_mu}")
        
        # Focus effect analysis
        focus_region = [(mu, fts) for mu, fts in results if abs(mu - focus.mu_focus) <= focus.sigma_focus]
        outside_region = [(mu, fts) for mu, fts in results if abs(mu - focus.mu_focus) > focus.sigma_focus]
        
        if focus_region and outside_region:
            focus_avg = np.mean([fts for _, fts in focus_region])
            outside_avg = np.mean([fts for _, fts in outside_region])
            print(f"   • Focus region average FTS: {focus_avg:.3f}")
            print(f"   • Outside region average FTS: {outside_avg:.3f}")
            
            sensitivity_ratio = outside_avg / focus_avg if focus_avg > 0 else 1.0
            if sensitivity_ratio > 1.2:
                print(f"   • Focus effect detected: {sensitivity_ratio:.1f}x sensitivity improvement")
    
    print(f"\n🎯 Integration Summary:")
    print(f"   • FTS focuses statistical power around μ={focus.mu_focus}")
    print(f"   • Lower FTS values indicate better consistency with focus prior")
    print(f"   • This example uses mock data - replace MockNLLCalculator with your likelihood function")
    
    print(f"\n📚 Next Steps:")
    print(f"   • See notebooks/FTS_plus.ipynb for full xRooFit integration")
    print(f"   • Use src/fts_core.py functions in your analysis")
    print(f"   • Replace focus function with your physics priors")


def show_integration_guide():
    """Show how to integrate with xRooFit analyses"""
    
    print("=" * 60)
    print("xRooFit Integration Guide") 
    print("=" * 60)
    
    integration_guide = """
Using FTS in your xRooFit analysis:

1. Import FTS modules:
   ```python
   from src.fts_core import fts_ts_obs, ProductionFocusFunction, CachedNLLCalculator
   ```

2. Create your focus function:
   ```python
   focus = ProductionFocusFunction(
       mu_focus=1.0,      # Center focus region
       sigma_focus=1.5,   # Focus width
       normalize=True
   )
   ```

3. Setup NLL calculator:
   ```python
   class XRooFitNLLCalculator:
       def get_nll_at_mu(self, dataset, mu):
           # Your xRooFit likelihood function evaluation
           return your_nll_function(mu)
   ```

4. Replace standard test statistic:
   ```python
   # Instead of: obs_ts = standard_likelihood_ratio(mu0)
   obs_ts = fts_ts_obs(nll_calc, "obsData", mu0, focus)
   ```

5. Use in HypoSpace:
   ```python
   hp.setObsTS(obs_ts, 0.0)  # Set observed FTS value
   ```

See notebooks/FTS_plus.ipynb for a complete working example.
    """
    print(integration_guide)

if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Run simple example
    simple_fts_example()
    
    # Show integration guide
    show_integration_guide()
    
    print(f"\n✨ Simple FTS integration example completed!")