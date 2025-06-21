import numpy as np
import matplotlib.pyplot as plt
import warnings

def safe_function_eval(func, x_vals):
    y_vals = np.full_like(x_vals, np.nan)
    undefined_indices = []
    
    for i, x in enumerate(x_vals):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                y = func(x)
                # Check for invalid results
                if np.isnan(y) or np.isinf(y):
                    undefined_indices.append(i)
                    y_vals[i] = 0.0
                else:
                    y_vals[i] = y
                    
        except (ZeroDivisionError, ValueError, RuntimeWarning, FloatingPointError):
            undefined_indices.append(i)
            y_vals[i] = 0.0
    
    return x_vals, y_vals, undefined_indices
    """
    Calculate definite integral using upper and lower Riemann sums.
    
    Parameters:
    -----------
    func : callable
        Function to integrate (should accept numpy arrays)
    a : float
        Lower bound of integration
    b : float
        Upper bound of integration
    n : int
        Number of subintervals
    method : str, optional
        'upper', 'lower', or 'both' (default: 'both')
    
    Returns:
    --------
    dict or float : Dictionary containing results if method='both', 
                   float value if method='upper' or 'lower'
        - 'lower_sum': Lower Riemann sum approximation
        - 'upper_sum': Upper Riemann sum approximation
        - 'average': Average of upper and lower sums
        - 'error_bound': Upper bound on approximation error
    """
    
def riemann_integration(func, a, b, n, method='both', handle_undefined='skip', min_valid_ratio=0.5):
    """
    Calculate definite integral using upper and lower Riemann sums with gap handling.
    
    Parameters:
    -----------
    func : callable
        Function to integrate (should accept numpy arrays)
    a : float
        Lower bound of integration
    b : float
        Upper bound of integration
    n : int
        Number of subintervals
    method : str, optional
        'upper', 'lower', or 'both' (default: 'both')
    handle_undefined : str, optional
        How to handle undefined function values:
        'skip' - skip intervals with undefined values
        'interpolate' - interpolate over undefined points
        'zero' - treat undefined values as zero
        'raise' - raise exception on undefined values
    min_valid_ratio : float, optional
        Minimum ratio of valid intervals required (default: 0.5)
    
    Returns:
    --------
    dict or float : Results with additional gap information
    """
    
    # Validate parameters
    if method not in ['upper', 'lower', 'both']:
        raise ValueError("Method must be 'upper', 'lower', or 'both'")
    
    if handle_undefined not in ['skip', 'interpolate', 'zero', 'raise']:
        raise ValueError("handle_undefined must be 'skip', 'interpolate', 'zero', or 'raise'")
    
    # Calculate width of each subinterval
    dx = (b - a) / n
    
    # Create array of x values at subinterval boundaries
    x_vals = np.linspace(a, b, n + 1)
    
    # Safely evaluate function at all points
    x_vals, y_vals, undefined_indices = safe_function_eval(func, x_vals, handle_undefined)
    
    # Track valid intervals and gaps
    valid_intervals = 0
    skipped_intervals = []
    gap_regions = []
    
    # Initialize sums
    lower_sum = 0
    upper_sum = 0
    
    # Process each interval
    for i in range(n):
        left_val = y_vals[i]
        right_val = y_vals[i + 1]
        
        # Check if interval is valid (both endpoints defined)
        if np.isnan(left_val) or np.isnan(right_val):
            if handle_undefined == 'skip':
                skipped_intervals.append(i)
                gap_regions.append([a + i * dx, a + (i + 1) * dx])
                continue
            elif handle_undefined == 'zero':
                left_val = 0 if np.isnan(left_val) else left_val
                right_val = 0 if np.isnan(right_val) else right_val
        
        valid_intervals += 1
        
        # Calculate min/max for this interval
        min_val = min(left_val, right_val)
        max_val = max(left_val, right_val)
        
        # Add to sums based on method
        if method in ['lower', 'both']:
            lower_sum += min_val * dx
        if method in ['upper', 'both']:
            upper_sum += max_val * dx
    
    # Check if we have enough valid intervals
    valid_ratio = valid_intervals / n
    if valid_ratio < min_valid_ratio:
        raise ValueError(f"Only {valid_ratio:.1%} of intervals are valid. "
                        f"Minimum required: {min_valid_ratio:.1%}")
    
    # Prepare results
    gap_info = {
        'total_intervals': n,
        'valid_intervals': valid_intervals,
        'skipped_intervals': len(skipped_intervals),
        'valid_ratio': valid_ratio,
        'gap_regions': gap_regions,
        'undefined_points': len(undefined_indices)
    }
    
    # Return based on method
    if method == 'lower':
        return {'value': lower_sum, 'gap_info': gap_info}
    elif method == 'upper':
        return {'value': upper_sum, 'gap_info': gap_info}
    else:  # method == 'both'
        average = (lower_sum + upper_sum) / 2
        error_bound = abs(upper_sum - lower_sum) / 2
        
        results = {
            'lower_sum': lower_sum,
            'upper_sum': upper_sum,
            'average': average,
            'error_bound': error_bound,
            'n_intervals': n,
            'interval': [a, b],
            'gap_info': gap_info
        }
        
        return results

def riemann_integration_refined(func, a, b, n, method='both'):
    """
    More accurate Riemann sum by sampling multiple points in each subinterval.
    
    Parameters:
    -----------
    func : callable
        Function to integrate
    a : float
        Lower bound
    b : float
        Upper bound
    n : int
        Number of subintervals
    method : str, optional
        'upper', 'lower', or 'both' (default: 'both')
    
    Returns:
    --------
    dict or float : Dictionary with refined upper and lower sums if method='both',
                   float value if method='upper' or 'lower'
    """
    
def riemann_integration_refined(func, a, b, n, method='both', handle_undefined='skip', min_valid_ratio=0.5):
    """
    More accurate Riemann sum by sampling multiple points in each subinterval with gap handling.
    
    Parameters:
    -----------
    func : callable
        Function to integrate
    a : float
        Lower bound
    b : float
        Upper bound
    n : int
        Number of subintervals
    method : str, optional
        'upper', 'lower', or 'both' (default: 'both')
    handle_undefined : str, optional
        How to handle undefined function values
    min_valid_ratio : float, optional
        Minimum ratio of valid intervals required
    
    Returns:
    --------
    dict or float : Dictionary with refined upper and lower sums and gap information
    """
    
    # Validate method parameter
    if method not in ['upper', 'lower', 'both']:
        raise ValueError("Method must be 'upper', 'lower', or 'both'")
    
    dx = (b - a) / n
    lower_sum = 0
    upper_sum = 0
    valid_intervals = 0
    skipped_intervals = []
    gap_regions = []
    
    # Process each interval
    for i in range(n):
        x_left = a + i * dx
        x_right = a + (i + 1) * dx
        
        # Sample multiple points in the subinterval
        sample_points = np.linspace(x_left, x_right, 10)
        _, y_samples, undefined_indices = safe_function_eval(func, sample_points, handle_undefined)
        
        # Check if interval has valid samples
        valid_samples = ~np.isnan(y_samples)
        valid_count = np.sum(valid_samples)
        
        if valid_count == 0:
            if handle_undefined == 'skip':
                skipped_intervals.append(i)
                gap_regions.append([x_left, x_right])
                continue
            elif handle_undefined == 'zero':
                min_val = max_val = 0
        elif valid_count < len(y_samples) * 0.5:  # Less than 50% valid samples
            if handle_undefined == 'skip':
                skipped_intervals.append(i)
                gap_regions.append([x_left, x_right])
                continue
            else:
                # Use only valid samples
                valid_y = y_samples[valid_samples]
                min_val = np.min(valid_y)
                max_val = np.max(valid_y)
        else:
            # Sufficient valid samples
            if handle_undefined == 'interpolate' or np.all(valid_samples):
                min_val = np.min(y_samples)
                max_val = np.max(y_samples)
            else:
                # Use only valid samples
                valid_y = y_samples[valid_samples]
                min_val = np.min(valid_y)
                max_val = np.max(valid_y)
        
        valid_intervals += 1
        
        # Add to sums based on method
        if method in ['lower', 'both']:
            lower_sum += min_val * dx
        if method in ['upper', 'both']:
            upper_sum += max_val * dx
    
    # Check if we have enough valid intervals
    valid_ratio = valid_intervals / n
    if valid_ratio < min_valid_ratio:
        raise ValueError(f"Only {valid_ratio:.1%} of intervals are valid. "
                        f"Minimum required: {min_valid_ratio:.1%}")
    
    # Prepare gap information
    gap_info = {
        'total_intervals': n,
        'valid_intervals': valid_intervals,
        'skipped_intervals': len(skipped_intervals),
        'valid_ratio': valid_ratio,
        'gap_regions': gap_regions
    }
    
    # Return based on method
    if method == 'lower':
        return {'value': lower_sum, 'gap_info': gap_info}
    elif method == 'upper':
        return {'value': upper_sum, 'gap_info': gap_info}
    else:  # method == 'both'
        average = (lower_sum + upper_sum) / 2
        error_bound = abs(upper_sum - lower_sum) / 2
        
        return {
            'lower_sum': lower_sum,
            'upper_sum': upper_sum,
            'average': average,
            'error_bound': error_bound,
            'n_intervals': n,
            'interval': [a, b],
            'gap_info': gap_info
        }

def visualize_riemann_sums(func, a, b, n, title="Riemann Sums Visualization"):
    """
    Visualize the upper and lower Riemann sums.
    
    Parameters:
    -----------
    func : callable
        Function to integrate
    a : float
        Lower bound
    b : float
        Upper bound
    n : int
        Number of subintervals
    title : str
        Plot title
    """
    
    # Calculate the sums
    result = riemann_integration_refined(func, a, b, n)
    
    # Create detailed plot
    x_plot = np.linspace(a, b, 1000)
    y_plot = func(x_plot)
    
    plt.figure(figsize=(12, 8))
    
    # Plot the function
    plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x)')
    
    # Draw rectangles for visualization
    dx = (b - a) / n
    for i in range(min(n, 20)):  # Limit to 20 rectangles for clarity
        x_left = a + i * dx
        x_right = a + (i + 1) * dx
        
        # Sample points in subinterval
        sample_points = np.linspace(x_left, x_right, 10)
        y_samples = func(sample_points)
        
        min_val = np.min(y_samples)
        max_val = np.max(y_samples)
        
        # Draw lower rectangle
        plt.bar(x_left + dx/2, min_val, dx, alpha=0.3, color='red', 
                edgecolor='red', label='Lower sum' if i == 0 else "")
        
        # Draw upper rectangle
        plt.bar(x_left + dx/2, max_val, dx, alpha=0.3, color='green', 
                edgecolor='green', label='Upper sum' if i == 0 else "")
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'{title}\nLower sum: {result["lower_sum"]:.6f}, Upper sum: {result["upper_sum"]:.6f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return result

# Example usage and test functions
if __name__ == "__main__":
    # Example 1: Simple quadratic function
    def f1(x):
        return x**2
    
    print("Example 1: f(x) = x² on [0, 2]")
    result1 = riemann_integration_refined(f1, 0, 2, 100)
    print(f"Lower sum: {result1['lower_sum']:.6f}")
    print(f"Upper sum: {result1['upper_sum']:.6f}")
    print(f"Average: {result1['average']:.6f}")
    print(f"Error bound: {result1['error_bound']:.6f}")
    print(f"Exact value: {8/3:.6f}")
    # Example with method parameter
    print("Testing method parameter:")
    print(f"Only lower sum: {riemann_integration_refined(f1, 0, 2, 100, method='lower'):.6f}")
    print(f"Only upper sum: {riemann_integration_refined(f1, 0, 2, 100, method='upper'):.6f}")
    print(f"Both sums: {riemann_integration_refined(f1, 0, 2, 100, method='both')}")
    print()
    
    # Example 2: Sine function
    def f2(x):
        return np.sin(x)
    
    print("Example 2: f(x) = sin(x) on [0, π]")
    result2 = riemann_integration_refined(f2, 0, np.pi, 100)
    print(f"Lower sum: {result2['lower_sum']:.6f}")
    print(f"Upper sum: {result2['upper_sum']:.6f}")
    print(f"Average: {result2['average']:.6f}")
    print(f"Error bound: {result2['error_bound']:.6f}")
    print(f"Exact value: {2:.6f}")
    print()
    
    # Example 3: Exponential function
    def f3(x):
        return np.exp(x)
    
    print("Example 3: f(x) = e^x on [0, 1]")
    result3 = riemann_integration_refined(f3, 0, 1, 100)
    print(f"Lower sum: {result3['lower_sum']:.6f}")
    print(f"Upper sum: {result3['upper_sum']:.6f}")
    print(f"Average: {result3['average']:.6f}")
    print(f"Error bound: {result3['error_bound']:.6f}")
    print(f"Exact value: {np.e - 1:.6f}")
    
    # Uncomment to see visualization
    # visualize_riemann_sums(f1, 0, 2, 10, "f(x) = x² on [0, 2]")