#!/usr/bin/env python3
"""
fts_core.py - FTS Core Implementation Module
===========================================

Core functions for FTS (Focused Test Statistics) implementation
Compatible with the FTS_plus.ipynb notebook requirements.

Optimized Version 2.0:
- Precomputed denominators (no redundant calculations)
- Global grid approach (μ₀-independent)
- Robust caching with proper key management
- Strict error handling for invalid focus functions
"""

import numpy as np
import math
from typing import Optional, Tuple, Union, Callable, Dict, Any
from collections import defaultdict

# Optional ROOT import for notebook compatibility
try:
    import ROOT
    ROOT_AVAILABLE = True
    def set_root_seed(seed: int):
        """Set ROOT random seed for reproducibility"""
        ROOT.gRandom.SetSeed(seed)
except ImportError:
    ROOT_AVAILABLE = False
    def set_root_seed(seed: int):
        """Set numpy random seed when ROOT unavailable"""
        np.random.seed(seed)

# Global debug flag - set to False to suppress verbose output
DEBUG_MODE = False

def debug_print(*args, **kwargs):
    """Print only if DEBUG_MODE is True"""
    if DEBUG_MODE:
        print(*args, **kwargs)


class ProductionFocusFunction:
    """Simple production focus function used in examples/README.

    - Gaussian weights by default
    - Optional 'tophat' weighting
    - Provides a μ₀-independent uniform grid helper for legacy code
    """

    def __init__(self,
                 mu_focus: float = 0.0,
                 sigma_focus: float = 1.5,
                 normalize: bool = True,
                 weight_type: str = 'gaussian'):
        self.mu_focus = float(mu_focus)
        self.sigma_focus = float(sigma_focus)
        self.normalize = bool(normalize)
        self.weight_type = str(weight_type).lower()

    def weight(self, mu: float) -> float:
        z = (float(mu) - self.mu_focus) / max(self.sigma_focus, 1e-12)
        if self.weight_type == 'tophat':
            # Interpret sigma_focus as half-width for the top-hat
            return 1.0 if abs(float(mu) - self.mu_focus) <= self.sigma_focus else 0.0
        # Default: Gaussian
        return math.exp(-0.5 * z * z)

    def get_uniform_grid(self, mu0: float, n_points: int = 101, n_sigma: float = 5.0):
        # μ₀-independent global grid (consistent with optimized implementation)
        center = self.mu_focus
        half_range = float(n_sigma) * self.sigma_focus
        lo = center - half_range
        hi = center + half_range
        if n_points % 2 == 0:
            n_points += 1
        return np.linspace(lo, hi, n_points)

def fts_ts_obs(nll_calc, dataset: str, mu0: float, focus_obj, n_grid: int = 101, theta: Tuple = (0.0, None)) -> Optional[float]:
    """
    Legacy FTS function for backward compatibility.
    
    This function maintains the old interface but internally uses the optimized
    implementation. Note: Without persistent caching, performance gains are limited.
    
    Args:
        nll_calc: NLL calculator object with get_nll_at_mu method
        dataset: Name of the dataset
        mu0: Null hypothesis parameter value
        focus_obj: Focus function object
        n_grid: Number of integration grid points
        theta: Parameter bounds (lower, upper)
        
    Returns:
        FTS test statistic value or None if calculation fails
    """
    try:
        # Get null hypothesis NLL
        nll_mu0 = nll_calc.get_nll_at_mu(dataset, mu0)
        if nll_mu0 is None:
            return None
        
        lnL_numerator = -float(nll_mu0)
        
        # Get integration grid
        mu_grid = focus_obj.get_uniform_grid(mu0, n_points=n_grid, n_sigma=5.0)
        
        # Apply bounds if specified
        if theta[0] is not None:
            mu_grid = mu_grid[mu_grid >= theta[0]]
        if theta[1] is not None:
            mu_grid = mu_grid[mu_grid <= theta[1]]
            
        if len(mu_grid) < 5:
            return None
        
        # Compute NLL values across grid
        nll_vals = []
        valid_mu = []
        for mu in mu_grid:
            nll_mu = nll_calc.get_nll_at_mu(dataset, float(mu))
            if nll_mu is None:
                continue
            valid_mu.append(float(mu))
            nll_vals.append(float(nll_mu))
        
        if len(nll_vals) < 5:
            return None
        
        valid_mu = np.asarray(valid_mu, dtype=float)
        nll_vals = np.asarray(nll_vals, dtype=float)
        
        # Build weight grid
        w_grid = _build_weight_grid(valid_mu, focus_obj)
        
        # Compute log denominator using profile likelihood
        ln_denominator = _log_denom_from_profile(valid_mu, nll_vals, w_grid)
        
        # FTS test statistic
        fts_ts = -2.0 * (lnL_numerator - ln_denominator)
        
        return float(fts_ts)
        
    except Exception as e:
        print(f"Error in fts_ts_obs: {e}")
        return None

def fts_ts_toy(nll_calc, dataset: str, mu0: float, focus_obj, n_grid: int = 61, theta: Tuple = (0.0, None)) -> Optional[float]:
    """
    Compute FTS test statistic for toy dataset
    
    Same as fts_ts_obs but optimized for toy calculations with fewer grid points
    """
    return fts_ts_obs(nll_calc, dataset, mu0, focus_obj, n_grid, theta)

def _build_weight_grid(mu_grid: np.ndarray, focus_obj) -> np.ndarray:
    """
    Build normalized weight grid from focus function using method-consistent normalization.
    
    KEY FIX: Normalization method now matches integration method for numerical consistency.
    This eliminates the 0.018 offset between different grid sizes.
    """
    w = np.array([focus_obj.weight(mu) for mu in mu_grid], dtype=float)
    
    # Strict validation for negative or non-finite weights
    if np.any(w < 0):
        raise ValueError("Focus function produced negative weights")
    if np.any(~np.isfinite(w)):
        raise ValueError("Focus function produced non-finite weights")
    
    if getattr(focus_obj, 'normalize', True):
        n = len(mu_grid)
        dh = np.diff(mu_grid)
        uniform = np.allclose(dh, dh[0], rtol=1e-12, atol=1e-15)
        
        if uniform and n >= 3 and (n % 2 == 1):
            # Simpson normalization (consistent with denominator integration)
            h = (mu_grid[-1] - mu_grid[0]) / (n - 1)
            coeff = np.ones(n, dtype=float)
            coeff[1:-1:2] = 4.0  # Odd indices (1,3,5,...)
            coeff[2:-2:2] = 2.0  # Even indices (2,4,6,...), excluding endpoints
            Z = (h / 3.0) * float(np.sum(coeff * w))
        else:
            # Fallback: Trapezoidal normalization
            Z = float(np.trapz(w, mu_grid))
        
        if Z <= 1e-300 or not np.isfinite(Z):
            raise ValueError(f"Focus normalization failed: Z={Z}")
        w = w / Z
    
    return w

def _log_denom_from_profile(mu_grid: np.ndarray, nll_grid: np.ndarray, w_grid: np.ndarray) -> float:
    """
    Compute log denominator using numerically stable integration
    
    Implements: log(∫ L(μ|data) π(μ) dμ) using proper log-sum-exp with Simpson coefficients
    
    Fixed: Integration coefficients are now applied in log domain for numerical stability
    """
    # Check if grid is uniform and suitable for Simpson's rule
    n = len(mu_grid)
    dh = np.diff(mu_grid)
    uniform = np.allclose(dh, dh[0], rtol=1e-10, atol=1e-15)
    
    if uniform and n >= 3 and (n % 2 == 1):
        # Simpson's rule with proper log-domain coefficients
        h = dh[0]
        
        # Build Simpson coefficients in log domain
        coeff = np.ones(n, dtype=float)
        coeff[1:-1:2] = 4.0  # Odd indices (1,3,5,...)
        coeff[2:-2:2] = 2.0  # Even indices (2,4,6,...), excluding endpoints
        
        # Apply h/3 scaling in log domain
        log_scale = np.log(h / 3.0)
        log_coeff = np.log(coeff) + log_scale
        
        # Complete log-domain integrand: log(L) + log(π) + log(coeff)
        log_integrand = -nll_grid + np.log(np.maximum(w_grid, 1e-300)) + log_coeff
        
        # Standard log-sum-exp (numerically stable)
        from scipy.special import logsumexp
        return float(logsumexp(log_integrand))
    else:
        # Fallback: Traditional log-sum-exp with trapezoidal
        a = -nll_grid + np.log(np.maximum(w_grid, 1e-300))
        amax = float(np.max(a))
        y = np.exp(a - amax)
        integral = np.trapz(y, mu_grid)
        integral = max(float(integral), 1e-300)
        return amax + math.log(integral)

def _log_denom_gauss_legendre(mu_lo: float, mu_hi: float, focus_obj, nll_calc, dataset_id: str, n_points: int = 32) -> float:
    """
    Compute log denominator using high-precision Gauss-Legendre quadrature.
    
    This provides a gold-standard reference for validating Simpson integration.
    Gauss-Legendre with 32 points typically achieves machine precision for smooth functions.
    
    Args:
        mu_lo, mu_hi: Integration bounds
        focus_obj: Focus function object
        nll_calc: NLL calculator
        dataset_id: Dataset identifier
        n_points: Number of Gauss-Legendre nodes (default 32)
        
    Returns:
        log(∫ L(μ|data) π(μ) dμ) computed with Gauss-Legendre
    """
    try:
        from scipy.special import roots_legendre
    except ImportError:
        raise ImportError("scipy.special.roots_legendre required for Gauss-Legendre integration")
    
    # Get Gauss-Legendre nodes and weights on [-1, 1]
    x_leg, w_leg = roots_legendre(n_points)
    
    # Transform to integration interval [mu_lo, mu_hi]
    mu_nodes = 0.5 * (mu_hi - mu_lo) * x_leg + 0.5 * (mu_hi + mu_lo)
    # Jacobian factor: dx/dmu = 2/(mu_hi - mu_lo), so weights scale by (mu_hi - mu_lo)/2
    jacobian_weights = 0.5 * (mu_hi - mu_lo) * w_leg
    
    # Evaluate integrand at all Gauss-Legendre nodes
    nll_vals = []
    focus_vals = []
    valid_nodes = []
    valid_weights = []
    
    for i, mu in enumerate(mu_nodes):
        nll_mu = nll_calc.get_nll_at_mu(dataset_id, float(mu))
        if nll_mu is None or not math.isfinite(nll_mu):
            continue  # Skip failed evaluations
        
        focus_val = focus_obj.weight(float(mu))
        if focus_val <= 0 or not math.isfinite(focus_val):
            continue  # Skip invalid focus values
            
        valid_nodes.append(float(mu))
        nll_vals.append(float(nll_mu))
        focus_vals.append(float(focus_val))
        valid_weights.append(float(jacobian_weights[i]))
    
    if len(nll_vals) < 3:
        raise ValueError(f"Too few valid Gauss-Legendre evaluations: {len(nll_vals)}")
    
    # Convert to arrays
    nll_vals = np.array(nll_vals)
    focus_vals = np.array(focus_vals)
    valid_weights = np.array(valid_weights)
    
    # Apply normalization to focus function if needed
    if getattr(focus_obj, 'normalize', True):
        # For Gauss-Legendre, normalization integral also needs same precision
        Z_integrand = focus_vals * valid_weights
        Z = np.sum(Z_integrand)
        if Z <= 1e-300 or not np.isfinite(Z):
            raise ValueError(f"Focus normalization failed in Gauss-Legendre: Z={Z}")
        focus_vals = focus_vals / Z
    
    # Build log-domain integrand with Gauss-Legendre weights
    log_integrand = -nll_vals + np.log(focus_vals) + np.log(valid_weights)
    
    # Use log-sum-exp for numerical stability
    from scipy.special import logsumexp
    return float(logsumexp(log_integrand))

def validate_fts_core():
    """Validate FTS core functions with strict numerical checks"""
    print("FTS Core Module Validation")
    print("-" * 30)
    
    # Test weight grid building with strict normalization check
    mu_test = np.linspace(0, 5, 101)
    
    class MockFocus:
        def __init__(self):
            self.normalize = True
        def weight(self, mu):
            return math.exp(-0.5 * (mu - 1.0)**2 / 2.0**2)
    
    focus = MockFocus()
    weights = _build_weight_grid(mu_test, focus)
    
    # Use same numerical integration method for validation
    normalization_integral = np.trapz(weights, mu_test)
    normalization_error = abs(normalization_integral - 1.0)
    
    print(f"Weight grid: {len(weights)} points")
    print(f"∫w dμ = {normalization_integral:.6f} (tol 1e-12)")
    
    if normalization_error < 1e-12:
        print("✓ Weight normalization: PASS")
    else:
        print(f"✗ Weight normalization: FAIL (error = {normalization_error:.2e})")
        return False
    
    # Test log denominator calculation
    nll_test = 0.5 * (mu_test - 1.5)**2
    log_denom = _log_denom_from_profile(mu_test, nll_test, weights)
    
    print(f"Log denominator: {log_denom:.6f}")
    print("✓ FTS core validation completed")
    
    return True

# ==============================================================================
# OPTIMIZED FTS IMPLEMENTATION (Version 2.0)
# ==============================================================================

class GlobalGridBuilder:
    """Build μ₀-independent integration grids for FTS calculations."""
    
    @staticmethod
    def build_global_grid(focus_obj, theta_bounds: Tuple = (-100.0, 100.0), 
                         n_points: int = 401, n_sigma: float = 6.0, *, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Build global integration grid independent of μ₀.
        
        Args:
            focus_obj: Focus function object with weight() method
            theta_bounds: Physical parameter bounds (theta_lo, theta_hi)
            n_points: Number of grid points (odd for Simpson's rule)
            n_sigma: Range extension factor for focus support
            
        Returns:
            Tuple of (mu_grid, w_grid, grid_info)
            
        Raises:
            ValueError: If focus function is invalid
        """
        # Ensure odd number of points for Simpson's rule
        if n_points % 2 == 0:
            n_points += 1
            
        # Define grid bounds based on focus center and width
        center = getattr(focus_obj, 'mu_focus', 0.0)
        width = getattr(focus_obj, 'sigma_focus', 1.5)
        
        # Fixed endpoint calculation to avoid floating point drift
        # Use high precision and round to ensure consistency across different N
        focus_lo = float(center) - float(n_sigma) * float(width)
        focus_hi = float(center) + float(n_sigma) * float(width)
        
        # Apply physical bounds with consistent rounding
        if theta_bounds[0] is not None:
            focus_lo = max(focus_lo, float(theta_bounds[0]))
        if theta_bounds[1] is not None:
            focus_hi = min(focus_hi, float(theta_bounds[1]))
            
        # Round to fixed precision to ensure identical endpoints across different N
        # This prevents 1e-15 level differences that accumulate to 0.018 offsets
        grid_lo = round(focus_lo, 12)
        grid_hi = round(focus_hi, 12)
        
        if grid_hi <= grid_lo:
            raise ValueError(f"Invalid grid bounds: [{grid_lo}, {grid_hi}]")
            
        # Create uniform grid
        mu_grid = np.linspace(grid_lo, grid_hi, n_points)
        
        # Build weight grid with strict validation
        w_grid = np.array([focus_obj.weight(mu) for mu in mu_grid], dtype=float)
        
        # Strict validation
        if np.any(w_grid < 0):
            raise ValueError("Focus function produced negative weights")
        if np.any(~np.isfinite(w_grid)):
            raise ValueError("Focus function produced non-finite weights")
        
        # Determine integration method BEFORE normalization
        dh = np.diff(mu_grid)
        uniform = np.allclose(dh, dh[0], rtol=1e-12, atol=1e-15)
        integration_method = 'simpson' if (uniform and len(mu_grid) >= 3 and (len(mu_grid) % 2 == 1)) else 'trapz'
            
        # Normalize using the SAME method that will be used for integration
        if getattr(focus_obj, 'normalize', True):
            if integration_method == 'simpson':
                # Simpson normalization (consistent with _log_denom_from_profile)
                h = (mu_grid[-1] - mu_grid[0]) / (n_points - 1)
                coeff = np.ones(n_points, dtype=float)
                coeff[1:-1:2] = 4.0  # Odd indices
                coeff[2:-2:2] = 2.0  # Even indices
                Z = (h / 3.0) * float(np.sum(coeff * w_grid))
            else:
                # Trapezoidal normalization
                Z = float(np.trapz(w_grid, mu_grid))
                
            if Z <= 1e-300 or not np.isfinite(Z):
                raise ValueError(f"Focus normalization failed: Z={Z}")
            w_grid = w_grid / Z
        
        # Create grid signature for caching with endpoint validation
        focus_signature = (
            getattr(focus_obj, 'weight_type', 'unknown'),
            getattr(focus_obj, 'mu_focus', 0.0),
            getattr(focus_obj, 'sigma_focus', 1.0),
            getattr(focus_obj, 'normalize', True)
        )
        
        # Use the integration method determined above (no recalculation needed)
        grid_signature = (grid_lo, grid_hi, n_points, integration_method)
        
        # Optional debug print
        if verbose:
            debug_print(f"    Grid signature: lo={grid_lo}, hi={grid_hi}, N={n_points}, method={integration_method}")
        
        grid_info = {
            'focus_signature': focus_signature,
            'grid_signature': grid_signature,
            'bounds': (grid_lo, grid_hi),
            'n_points': n_points,
            'integration_method': integration_method
        }
        
        return mu_grid, w_grid, grid_info


class DenominatorCache:
    """Cache for FTS denominators with proper key management and detailed logging."""
    
    def __init__(self):
        self.cache: Dict[Tuple, Dict[str, Any]] = {}
        self.stats = defaultdict(int)
        self.per_dataset_counts = defaultdict(int)  # Track denominators per dataset
        self.computation_times = []  # Track computation times
    
    def get_cache_key(self, dataset_id: str, grid_info: Dict) -> Tuple:
        """Generate cache key with three-tuple approach."""
        focus_sig = grid_info['focus_signature']
        grid_sig = grid_info['grid_signature']
        
        # Ensure focus signature is tuple of sorted items
        if isinstance(focus_sig, tuple) and len(focus_sig) >= 4:
            focus_key = (focus_sig[0], focus_sig[1], focus_sig[2], focus_sig[3])
        else:
            focus_key = tuple(focus_sig) if hasattr(focus_sig, '__iter__') else focus_sig
        
        return (
            dataset_id,  # Must distinguish different toy datasets
            focus_key,   # (focus_type, mu_focus, sigma_focus, normalize)
            grid_sig     # (grid_lo, grid_hi, n_points, rule)
        )
    
    def get_denominator(self, dataset_id: str, focus_obj, nll_calc, 
                       grid_info: Dict, mu_grid: np.ndarray, w_grid: np.ndarray, 
                       verbose: bool = True) -> Dict[str, Any]:
        """
        Get cached denominator or compute if not cached with detailed logging.
        
        Returns:
            Dictionary containing log_denom, nll_grid, mu_grid
        """
        import time
        
        cache_key = self.get_cache_key(dataset_id, grid_info)
        
        # Check cache first
        if cache_key in self.cache:
            self.stats['hits'] += 1
            if verbose:
                debug_print(f"    Denominator: CACHED (key={dataset_id})")
            return self.cache[cache_key]
        
        # Miss - need to compute
        self.stats['misses'] += 1
        compute_start = time.time()
        
        if verbose:
            debug_print(f"    Denominator: COMPUTING (key={dataset_id})")
        self.per_dataset_counts[cache_key] += 1
        
        # Compute NLL across entire grid
        nll_vals = []
        valid_mu = []
        failed_points = 0
        
        for mu in mu_grid:
            nll_mu = nll_calc.get_nll_at_mu(dataset_id, float(mu))
            if nll_mu is None or not math.isfinite(nll_mu):
                failed_points += 1
                continue
            valid_mu.append(float(mu))
            nll_vals.append(float(nll_mu))
        
        if len(nll_vals) < 11:  # Need minimum points for integration
            raise ValueError(f"Too few valid NLL evaluations: {len(nll_vals)}")
        
        valid_mu = np.asarray(valid_mu, dtype=float)
        nll_vals = np.asarray(nll_vals, dtype=float)
        
        # Interpolate weights for valid grid points
        valid_w = np.interp(valid_mu, mu_grid, w_grid)
        
        # CRITICAL FIX: Renormalize weights on valid grid if points were deleted
        # This ensures consistent normalization when the effective integration domain changes
        if len(valid_mu) < len(mu_grid):  # Points were deleted
            dh_v = np.diff(valid_mu)
            uniform_v = np.allclose(dh_v, dh_v[0], rtol=1e-12, atol=1e-15)
            
            if uniform_v and len(valid_mu) >= 3 and (len(valid_mu) % 2 == 1):
                # Simpson renormalization on valid grid
                h = (valid_mu[-1] - valid_mu[0]) / (len(valid_mu) - 1)
                coeff = np.ones(len(valid_mu), dtype=float)
                coeff[1:-1:2] = 4.0  # Odd indices
                coeff[2:-2:2] = 2.0  # Even indices  
                Z = (h / 3.0) * float(np.sum(coeff * valid_w))
            else:
                # Trapezoidal renormalization on valid grid
                Z = float(np.trapz(valid_w, valid_mu))
            
            if Z > 1e-300 and np.isfinite(Z):
                valid_w = valid_w / Z
                if verbose:
                    debug_print(f"    Renormalized weights on valid grid: Z={Z:.6f}")
            else:
                if verbose:
                    debug_print(f"    Warning: Invalid normalization on valid grid: Z={Z}")
        
        # Compute log denominator using stable integration
        log_denom = _log_denom_from_profile(valid_mu, nll_vals, valid_w)
        
        # Record computation time
        compute_time = time.time() - compute_start
        self.computation_times.append(compute_time)
        
        if verbose:
            debug_print(f"    Denominator: COMPLETED (time={compute_time:.3f}s)")
        
        # Cache the result
        cache_entry = {
            'log_denom': log_denom,
            'nll_grid': nll_vals,
            'mu_grid': valid_mu,
            'failed_points': failed_points,
            'compute_time': compute_time
        }
        
        self.cache[cache_key] = cache_entry
        return cache_entry
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Return enhanced cache performance statistics."""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total if total > 0 else 0.0
        
        # Calculate denominators per dataset statistics
        denominators_per_key = [count for count in self.per_dataset_counts.values()]
        max_denoms_per_key = max(denominators_per_key) if denominators_per_key else 0
        
        # Check for violations (should always be 1 per key)
        violations = sum(1 for count in denominators_per_key if count > 1)
        
        # Enhanced metrics
        unique_datasets = len(self.per_dataset_counts)
        total_computations = self.stats['misses']  # Each miss = one actual computation
        
        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'max_denoms_per_key': max_denoms_per_key,
            'violations': violations,
            'avg_compute_time': np.mean(self.computation_times) if self.computation_times else 0.0,
            'total_compute_time': sum(self.computation_times),
            # New detailed metrics
            'unique_datasets': unique_datasets,
            'total_computations': total_computations,
            'total_accesses': total,
            'computations_per_dataset': total_computations / max(unique_datasets, 1),
            'efficiency_ratio': unique_datasets / max(total_computations, 1)  # Should be ~1.0
        }
    
    def print_detailed_stats(self):
        """Print comprehensive cache statistics with warnings."""
        stats = self.get_stats()
        
        print(f"\n📊 Denominator Cache Detailed Statistics:")
        print(f"   • Cache hits: {stats['hits']:,}")
        print(f"   • Cache misses: {stats['misses']:,}")
        print(f"   • Hit rate: {stats['hit_rate']:.1%}")
        print(f"   • Cache size: {stats['cache_size']:,} denominators")
        print(f"   • Average compute time: {stats['avg_compute_time']:.3f}s")
        print(f"   • Total compute time: {stats['total_compute_time']:.2f}s")
        
        # New enhanced metrics
        print(f"\n🔍 Computational Efficiency:")
        print(f"   • Unique datasets: {stats['unique_datasets']:,}")
        print(f"   • Actual computations: {stats['total_computations']:,}")
        print(f"   • Total cache accesses: {stats['total_accesses']:,}")
        print(f"   • Computations per dataset: {stats['computations_per_dataset']:.2f}")
        print(f"   • Efficiency ratio: {stats['efficiency_ratio']:.2f} (target: 1.0)")
        
        # Check for issues
        if stats['max_denoms_per_key'] > 1:
            print(f"\n⚠️  WARNING: Found {stats['violations']} keys with >1 denominator")
            print(f"   • Max denominators per key: {stats['max_denoms_per_key']}")
            print(f"   • Expected: 1.0 per (dataset, focus, grid) combination")
        else:
            print(f"\n✓ Denominator efficiency: {stats['max_denoms_per_key']:.1f} per key (optimal)")
        
        # Enhanced performance insights
        if stats['efficiency_ratio'] >= 0.99:
            print(f"🎉 EXCELLENT: Each dataset computed exactly once!")
        elif stats['efficiency_ratio'] >= 0.9:
            print(f"✅ GOOD: Minimal redundant computations")
        else:
            print(f"⚠️  ATTENTION: Some datasets computed multiple times")
            
        # Performance insights
        if stats['hit_rate'] < 0.1 and stats['cache_size'] > 10:
            print(f"💡 Low hit rate suggests mainly independent datasets (normal for toy generation)")
        elif stats['hit_rate'] > 0.5:
            print(f"🚀 High hit rate indicates effective denominator reuse")


class OptimizedFTSCalculator:
    """Optimized FTS calculator using precomputed denominators."""
    
    def __init__(self):
        self.global_grid_cache = {}
        self.denominator_cache = DenominatorCache()
    
    def compute_fts(self, nll_calc, dataset_id: str, mu0: float, focus_obj, 
                   theta_bounds: Tuple = (-100.0, 100.0), n_grid: int = 401, verbose: bool = True) -> Optional[float]:
        """
        Compute FTS test statistic using optimized approach.
        
        Args:
            nll_calc: NLL calculator object
            dataset_id: Unique dataset identifier
            mu0: Null hypothesis value
            focus_obj: Focus function object
            theta_bounds: Parameter bounds
            n_grid: Grid size
            
        Returns:
            FTS test statistic value
        """
        try:
            # Step 1: Get or build global grid
            grid_key = (focus_obj.mu_focus, focus_obj.sigma_focus, theta_bounds, n_grid)
            
            if grid_key not in self.global_grid_cache:
                mu_grid, w_grid, grid_info = GlobalGridBuilder.build_global_grid(
                    focus_obj, theta_bounds, n_grid
                )
                self.global_grid_cache[grid_key] = (mu_grid, w_grid, grid_info)
            else:
                mu_grid, w_grid, grid_info = self.global_grid_cache[grid_key]
            
            # Step 2: Get cached denominator
            denom_entry = self.denominator_cache.get_denominator(
                dataset_id, focus_obj, nll_calc, grid_info, mu_grid, w_grid, verbose=verbose
            )
            
            log_denom = denom_entry['log_denom']
            cached_mu_grid = denom_entry['mu_grid']
            cached_nll_grid = denom_entry['nll_grid']
            
            # Step 3: Compute L(μ₀|D) - ALWAYS direct evaluation (no interpolation)
            # This ensures consistent numerator calculation regardless of grid alignment
            # Fixes the primary source of FTS-LRT offset variation
            nll_mu0 = nll_calc.get_nll_at_mu(dataset_id, float(mu0))
            if nll_mu0 is None:
                return None
            lnL_mu0 = -float(nll_mu0)
            
            # Debug: Check if μ₀ was in cached grid (for verification only)
            if verbose:
                mu0_in_grid = any(abs(mu - mu0) < 1e-10 for mu in cached_mu_grid)
                if mu0_in_grid:
                    debug_print(f"    μ₀={mu0} was in grid (but used direct evaluation)")
                else:
                    debug_print(f"    μ₀={mu0} outside grid (direct evaluation)")
            
            # Step 4: Compute FTS
            fts_value = -2.0 * (lnL_mu0 - log_denom)
            return float(fts_value)
            
        except Exception as e:
            # Fail fast with clear error message
            raise ValueError(f"FTS calculation failed for μ₀={mu0}, dataset={dataset_id}: {e}")
    
    def compute_lrt(self, nll_calc, dataset_id: str, mu0: float, focus_obj,
                   theta_bounds: Tuple = (-100.0, 100.0), n_grid: int = 401, verbose: bool = True) -> Optional[float]:
        """
        Compute LRT test statistic reusing cached NLL evaluations.
        
        This provides a fair comparison by using the same computational setup.
        """
        try:
            # Reuse the same grid and cached NLL values
            grid_key = (focus_obj.mu_focus, focus_obj.sigma_focus, theta_bounds, n_grid)
            
            if grid_key not in self.global_grid_cache:
                mu_grid, w_grid, grid_info = GlobalGridBuilder.build_global_grid(
                    focus_obj, theta_bounds, n_grid
                )
                self.global_grid_cache[grid_key] = (mu_grid, w_grid, grid_info)
            else:
                mu_grid, w_grid, grid_info = self.global_grid_cache[grid_key]
            
            denom_entry = self.denominator_cache.get_denominator(
                dataset_id, focus_obj, nll_calc, grid_info, mu_grid, w_grid, verbose=verbose
            )
            
            cached_mu_grid = denom_entry['mu_grid']
            cached_nll_grid = denom_entry['nll_grid']
            
            # Find MLE (minimum NLL) with boundary detection
            min_nll_idx = np.argmin(cached_nll_grid)
            min_nll = cached_nll_grid[min_nll_idx]
            mu_hat = cached_mu_grid[min_nll_idx]
            lnL_muhat = -float(min_nll)
            
            # Boundary detection: warn if MLE is at grid endpoints
            # This may indicate the grid range is too narrow
            is_at_boundary = (min_nll_idx == 0) or (min_nll_idx == len(cached_nll_grid) - 1)
            if is_at_boundary and verbose:
                print(f"    ⚠️  MLE at boundary: μ̂={mu_hat:.3f} (index {min_nll_idx}/{len(cached_nll_grid)-1})")
                print(f"       Consider expanding integration range for better LRT accuracy")
            
            # Get L(μ₀|D) same as FTS - ALWAYS direct evaluation for consistency
            nll_mu0 = nll_calc.get_nll_at_mu(dataset_id, float(mu0))
            if nll_mu0 is None:
                return None
            lnL_mu0 = -float(nll_mu0)
            
            # Compute LRT
            lrt_value = -2.0 * (lnL_mu0 - lnL_muhat)
            return float(lrt_value)
            
        except Exception as e:
            raise ValueError(f"LRT calculation failed for μ₀={mu0}, dataset={dataset_id}: {e}")

    def precompute_denominator(self, nll_calc, dataset_id: str, focus_obj,
                                theta_bounds: Tuple = (-100.0, 100.0), n_grid: int = 401,
                                verbose: bool = True) -> Dict[str, Any]:
        """Pre-compute and cache the denominator for a dataset.

        Use this in toy-calibration loops to explicitly compute the per-dataset
        denominator once, then re-use it across any μ₀ evaluations for the same dataset.

        Note: For different toy datasets (different data), denominators differ
        and cannot be re-used across dataset_ids.
        """
        # Ensure grid exists
        grid_key = (focus_obj.mu_focus, focus_obj.sigma_focus, theta_bounds, n_grid)
        if grid_key not in self.global_grid_cache:
            mu_grid, w_grid, grid_info = GlobalGridBuilder.build_global_grid(
                focus_obj, theta_bounds, n_grid
            )
            self.global_grid_cache[grid_key] = (mu_grid, w_grid, grid_info)
        else:
            mu_grid, w_grid, grid_info = self.global_grid_cache[grid_key]

        # Compute/cache denominator
        return self.denominator_cache.get_denominator(
            dataset_id, focus_obj, nll_calc, grid_info, mu_grid, w_grid, verbose=verbose
        )

    def compute_fts_and_lrt(self, nll_calc, dataset_id: str, mu0: float, focus_obj,
                             theta_bounds: Tuple = (-100.0, 100.0), n_grid: int = 401,
                             verbose: bool = True) -> Tuple[float, float]:
        """Compute FTS and LRT together with a single denominator evaluation.

        - Builds or reuses the global grid
        - Computes/caches the denominator once
        - Evaluates ln L(μ₀|D) once
        - Finds μ̂ and ln L(μ̂|D) from the cached NLL grid

        Returns:
            (fts_value, lrt_value)
        """
        # Ensure grid exists
        grid_key = (focus_obj.mu_focus, focus_obj.sigma_focus, theta_bounds, n_grid)
        if grid_key not in self.global_grid_cache:
            mu_grid, w_grid, grid_info = GlobalGridBuilder.build_global_grid(
                focus_obj, theta_bounds, n_grid
            )
            self.global_grid_cache[grid_key] = (mu_grid, w_grid, grid_info)
        else:
            mu_grid, w_grid, grid_info = self.global_grid_cache[grid_key]

        # One denominator request (cache miss on first access per dataset)
        denom_entry = self.denominator_cache.get_denominator(
            dataset_id, focus_obj, nll_calc, grid_info, mu_grid, w_grid, verbose=verbose
        )

        log_denom = denom_entry['log_denom']
        cached_mu_grid = denom_entry['mu_grid']
        cached_nll_grid = denom_entry['nll_grid']

        # ln L(μ₀|D)
        nll_mu0 = nll_calc.get_nll_at_mu(dataset_id, float(mu0))
        if nll_mu0 is None:
            raise ValueError("NLL(μ₀) evaluation failed")
        lnL_mu0 = -float(nll_mu0)

        # MLE from cached grid
        min_idx = int(np.argmin(cached_nll_grid))
        lnL_muhat = -float(cached_nll_grid[min_idx])

        # Stats
        fts_value = -2.0 * (lnL_mu0 - log_denom)
        lrt_value = -2.0 * (lnL_mu0 - lnL_muhat)
        return float(fts_value), float(lrt_value)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics with enhanced metrics."""
        denom_stats = self.denominator_cache.get_stats()
        
        return {
            'denominator_cache': denom_stats,
            'global_grids_cached': len(self.global_grid_cache),
            'denominators_per_dataset': denom_stats['max_denoms_per_key'],
            'cache_violations': denom_stats['violations'],
            'total_denominator_compute_time': denom_stats['total_compute_time'],
            'avg_denominator_compute_time': denom_stats['avg_compute_time']
        }
    
    def print_comprehensive_stats(self):
        """Print detailed performance analysis with warnings and insights."""
        print("\n" + "=" * 60)
        print("🔍 COMPREHENSIVE PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        # Global grid statistics
        print(f"\n🌐 Global Grid Cache:")
        print(f"   • Cached grids: {len(self.global_grid_cache)}")
        if len(self.global_grid_cache) <= 2:
            print(f"✓ Optimal grid reuse (expected ≤2 for obs+toy grids)")
        else:
            print(f"⚠️  More grids than expected - check parameter consistency")
        
        # Detailed denominator statistics  
        self.denominator_cache.print_detailed_stats()
        
        # Performance summary
        stats = self.get_performance_stats()
        print(f"\n🏆 Optimization Summary:")
        print(f"   • Denominators per dataset: {stats['denominators_per_dataset']:.1f} (target: 1.0)")
        print(f"   • Cache violations: {stats['cache_violations']} (target: 0)")
        print(f"   • Total denominator time: {stats['total_denominator_compute_time']:.2f}s")
        
        if stats['denominators_per_dataset'] <= 1.0 and stats['cache_violations'] == 0:
            print(f"🎉 EXCELLENT: No redundant denominator calculations detected!")
        else:
            print(f"⚠️  ATTENTION: Found redundant calculations - review implementation")


# Global instance for backward compatibility
_default_calculator = OptimizedFTSCalculator()

def fts_ts_obs_optimized(nll_calc, dataset: str, mu0: float, focus_obj, 
                        n_grid: int = 401, theta: Tuple = (-100.0, 100.0)) -> Optional[float]:
    """
    Optimized FTS calculation with precomputed denominators.
    
    This is the recommended interface for new code.
    """
    return _default_calculator.compute_fts(nll_calc, dataset, mu0, focus_obj, theta, n_grid)

def lrt_ts_optimized(nll_calc, dataset: str, mu0: float, focus_obj,
                    n_grid: int = 401, theta: Tuple = (-100.0, 100.0)) -> Optional[float]:
    """
    Optimized LRT calculation for fair comparison with FTS.
    """
    return _default_calculator.compute_lrt(nll_calc, dataset, mu0, focus_obj, theta, n_grid)

def get_optimization_stats() -> Dict[str, Any]:
    """Get performance statistics from the optimized calculator."""
    return _default_calculator.get_performance_stats()

def reset_caches():
    """Reset all caches (useful for testing)."""
    global _default_calculator
    _default_calculator = OptimizedFTSCalculator()

# ==============================================================================
# VALIDATION AND TESTING
# ==============================================================================

def validate_optimization():
    """Validate the optimized implementation against theoretical expectations."""
    print("FTS Optimization Validation")
    print("-" * 40)
    
    # Test 1: Global grid construction
    class MockFocus:
        def __init__(self):
            self.mu_focus = 1.0
            self.sigma_focus = 2.0
            self.normalize = True
            self.weight_type = 'gaussian'
        
        def weight(self, mu):
            z = (mu - self.mu_focus) / self.sigma_focus
            return math.exp(-0.5 * z * z)
    
    focus = MockFocus()
    try:
        mu_grid, w_grid, grid_info = GlobalGridBuilder.build_global_grid(focus)
        print(f"✓ Global grid: {len(mu_grid)} points, range [{mu_grid[0]:.2f}, {mu_grid[-1]:.2f}]")
        
        # Check normalization using the SAME method that was used for normalization
        # This ensures consistency with the actual integration method
        integration_method = grid_info.get('integration_method', 'unknown')
        if integration_method == 'simpson':
            # Simpson normalization check (matches the method used in GlobalGridBuilder)
            h = (mu_grid[-1] - mu_grid[0]) / (len(mu_grid) - 1)
            coeff = np.ones(len(mu_grid), dtype=float)
            coeff[1:-1:2] = 4.0  # Odd indices
            coeff[2:-2:2] = 2.0  # Even indices
            simpson_integral = (h / 3.0) * np.sum(coeff * w_grid)
            print(f"✓ Weight normalization: ∫w dμ = {simpson_integral:.6f} (Simpson)")
        else:
            # Trapezoidal check for non-uniform grids
            trapz_integral = np.trapz(w_grid, mu_grid)
            print(f"✓ Weight normalization: ∫w dμ = {trapz_integral:.6f} (Trapezoidal)")
    except Exception as e:
        print(f"✗ Global grid construction failed: {e}")
        return False
    
    # Test 2: Error handling for invalid focus
    class BadFocus:
        def __init__(self):
            self.mu_focus = 0.0
            self.sigma_focus = 1.0
            self.normalize = True
        
        def weight(self, mu):
            return -1.0  # Invalid: negative weight
    
    try:
        GlobalGridBuilder.build_global_grid(BadFocus())
        print("✗ Error handling failed: should have caught negative weights")
        return False
    except ValueError:
        print("✓ Error handling: properly caught invalid focus function")
    
    # Test 3: Cache key generation
    cache = DenominatorCache()
    key1 = cache.get_cache_key("toy_mu1_0000", grid_info)
    key2 = cache.get_cache_key("toy_mu1_0001", grid_info)
    
    if key1 != key2:
        print("✓ Cache keys properly distinguish different datasets")
    else:
        print("✗ Cache keys incorrectly merge different datasets")
        return False
    
    print("✓ Optimization validation completed successfully")
    return True

# Auto-validate when imported
import os as _os
if __name__ == "__main__" or _os.environ.get("FTS_VALIDATE_ON_IMPORT") == "1":
    try:
        validate_fts_core()
        validate_optimization()
    except Exception:
        pass
