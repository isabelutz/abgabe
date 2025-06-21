import numpy as np
import warnings 
class Integration:
    def __init__(self):
        pass
    def find_gaps_mathfunc(self,x_vals,math_func):
         y_vals = np.full_like(x_vals, np.nan)
         undefined_indices = []
         for i, x in enumerate(x_vals):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    y = math_func(x)
                    if np.isnan(y) or np.isinf(y):
                        undefined_indices.append(i)
                        y_vals[i] = 0.0
                    else:
                        y_vals[i] = y
            except (ZeroDivisionError, ValueError, RuntimeWarning, FloatingPointError):
                undefined_indices.append(i)
                y_vals[i] = 0.0
         return x_vals, y_vals, undefined_indices

    def mathematic_func(self, math_operation):
        return math_operation 
    def up_lowsums(self, math_func, low_interv, up_interv, numberintervall):
        width_rectangle = (up_interv - low_interv) / numberintervall # Berechnet die Breite der einzelnen Intervallrechtecke
        x_vals = np.linspace(low_interv, up_interv, numberintervall + 1) # definiert die 4 Punkte am Ende jedes intervalls (x-Werte)
        y_vals = math_func(x_vals) # Berechnet die jeweiligen Werte für jedes Intervall (y-Werte)
        lower_sum = 0
        upper_sum = 0
        for i in range(numberintervall):
            left_val = y_vals[i] # Wert der Funktion am linken Intervall
            right_val = y_vals[i + 1] # Wert der Funktion am rechten Intervall
            min_val = min(left_val, right_val) # Aussuchen des niedrigeren Intervalls
            max_val = max(left_val, right_val) # Aussuchen des höheren Intervalls
            lower_sum += min_val * width_rectangle # Berechnung der Untersumme und Addition zu den bisherigen Untersummen
            upper_sum += max_val * width_rectangle # Berechnung der Obersumme und Addition zu den bisherigen Obersummen

        average = (lower_sum + upper_sum) / 2 # Durchschnitt zwischen Ober und Untersumme
        return {
        'lower_sum': lower_sum,
        'upper_sum': upper_sum,
        'average': average}
    
   
    def trapez_rule(self):
        pass
    def montecarlo_simulation(self):
        pass
    def visualize_method(self):
        pass
    def test_all(self): # allg. Methode die alle mit best. Parameter testet
        self.up_lowsums()
        self.trapez_rule()
        self.montecarlo_simulation()
    def test_impossible(self):
        self.up_lowsums() # 2 Intervalle
        self.up_lowsums() # 4 Intervalle
        self.up_lowsums() # 8 Intervalle
        self.up_lowsums() # 16 Intervalle

def riemann_integration(func, a, b, n, method='both'):
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
    dict : Dictionary containing results
        - 'lower_sum': Lower Riemann sum approximation
        - 'upper_sum': Upper Riemann sum approximation
        - 'average': Average of upper and lower sums
        - 'error_bound': Upper bound on approximation error
    """
    
    # Calculate width of each subinterval
    dx = (b - a) / n
    
    # Create array of x values at subinterval boundaries
    x_vals = np.linspace(a, b, n + 1)
    
    # Calculate function values at all points
    y_vals = func(x_vals)
    
    # Calculate lower sum (minimum value in each subinterval)
    lower_sum = 0
    upper_sum = 0
    
    for i in range(n):
        # Get function values at left and right endpoints of subinterval
        left_val = y_vals[i]
        right_val = y_vals[i + 1]
        
        # For monotonic functions, min/max are at endpoints
        # For general case, we approximate using endpoint values
        min_val = min(left_val, right_val)
        max_val = max(left_val, right_val)
        
        lower_sum += min_val * dx
        upper_sum += max_val * dx
    
    # Calculate average and error bound
    average = (lower_sum + upper_sum) / 2
    error_bound = abs(upper_sum - lower_sum) / 2
    
    results = {
        'lower_sum': lower_sum,
        'upper_sum': upper_sum,
        'average': average,
        'error_bound': error_bound,
        'n_intervals': n,
        'interval': [a, b]
    }
    
    return results