import numpy as np
import astropy.constants as const

# Constantes physiques
h = const.h.value
c = const.c.value
k_B = const.k_B.value
sigma = const.sigma_sb.value
b = 2.898e-3  # Constante de Wien en m·K

# Calcul de la luminosité (L = 4πR²σT⁴)
def stefan_law(T, R):
    """Calcule la luminosité totale d'un corps noir."""
    return 4 * np.pi * R**2 * sigma * T**4

# Calcul du pic d'émission : λ_max = b / T (b = 2.898e-3 m·K)
def wien_law(T):
    """Calcule la longueur d'onde du maximum d'émission."""
    return b / T

# Spectre du corps noir (simplifié)
def planck_law(lam, T):
    """
    Renvoie la luminance spectrale B_lambda(T)
    lam : longueur d'onde en m
    T   : température en K
    """
    return (2 * h * c**2) / (lam**5 * (np.exp((h * c) / (lam * k_B * T)) - 1))

# Spectre du corps noir (simplifié)
def planck_law(lam, T):
    """
    Renvoie la luminance spectrale B_lambda(T)
    lam : longueur d'onde en m
    T   : température en K
    """
    return (2 * h * c**2) / (lam**5 * (np.exp((h * c) / (lam * k_B * T)) - 1))