import numpy as np

def luminosite_transit(t, t_start, t_end, R_star, R_planet):
    """ Simule la courbe de lumière d'une étoile subissant un transit d'exoplanète.

    Parameters
    ----------
    t : array-like
        Temps en jours.
    t_start : float
        Temps de début du transit en jours.
    t_end : float
        Temps de fin du transit en jours.
    R_star : float
        Rayon de l'étoile en m.
    R_planet : float
        Rayon de la planète en m.

    Returns
    ----------
    lightcurve : array-like
        Courbe de lumière simulée.
    """
    # Condition de transit
    transit_condition = (t >= t_start) & (t <= t_end)

    # Flux de l'étoile sans transit normalisé à 1
    flux_etoile = 1.0

    # Flux du transit
    K = (R_planet / R_star)**2
    flux_transit = flux_etoile * (1 - K)
    # Utilisation de np.where pour créer la courbe de lumière
    lightcurve = np.where(transit_condition, flux_transit, flux_etoile)

    return lightcurve

def incertitude_rayon_planete(temps, t_start, t_end, flux, R_etoile):
    """ Calcule l'incertitude sur le rayon de la planète à partir de la courbe de lumière bruitée.

    Parameters
    ----------
    temps : array-like
        Temps en jours.
    t_start : float
        Temps de début du transit en jours.
    t_end : float
        Temps de fin du transit en jours.
    flux : array-like
        Flux bruité.
    R_etoile : float
        Rayon de l'étoile en m.

    Returns
    ----------
    std_rayon_planete : float
        Incertitude sur le rayon de la planète en m.
    """

    transit_mask = (temps >= t_start) & (temps <= t_end)
    hors_transit_mask = ~transit_mask

    flux_transit_bruitee = flux[transit_mask]
    flux_hors_transit_bruitee = flux[hors_transit_mask]

    mean_transit = np.mean(flux_transit_bruitee)
    mean_hors_transit = np.mean(flux_hors_transit_bruitee)

    std_transit = np.std(flux_transit_bruitee)
    std_hors_transit = np.std(flux_hors_transit_bruitee)

    delta = 1 - mean_transit / mean_hors_transit
    if delta <= 0:
        raise ValueError("La différence entre les flux moyens est invalide (delta <= 0).")

    # Propagation d'incertitude
    partial_transit = -(R_etoile / (2 * np.sqrt(delta) * mean_hors_transit))
    partial_hors_transit = (R_etoile * mean_transit) / (2 * np.sqrt(delta) * mean_hors_transit**2)

    std_rayon_planete = np.sqrt(
        (partial_transit * std_transit)**2 + (partial_hors_transit * std_hors_transit)**2
    )

    return std_rayon_planete

def calcul_incertitude_rayon(temps, t_start, t_end, flux, R_etoile, bin_size=1):
    """
    Calcule l'incertitude sur le rayon de la planète en moyennant les points par bin de temps.

    Parameters
    ----------
    temps : array-like
        Tableau des temps.
    flux : array-like
        Tableau des flux.
    t_start : float
        Temps de début du transit.
    t_end : float
        Temps de fin du transit.
    R_etoile : float
        Rayon de l'étoile en mètres.
    bin_size : float
        Taille des bins en jours.

    Returns
    ----------
    incertitude : float
        Incertitude sur le rayon de la planète.
    """
    # Définir les bins
    bins = np.arange(temps.min(), temps.max() + bin_size, bin_size)
    indices = np.digitize(temps, bins)

    # Moyennage des flux par bin
    temps_binned = []
    flux_binned = []
    for i in range(1, len(bins)):
        mask = indices == i
        if np.any(mask):  # Vérifier si le bin contient des données
            temps_binned.append(np.mean(temps[mask]))
            flux_binned.append(np.mean(flux[mask]))

    temps_binned = np.array(temps_binned)
    flux_binned = np.array(flux_binned)

    # Calcul de l'incertitude avec les données moyennées
    incertitude = incertitude_rayon_planete(temps_binned, t_start, t_end, flux_binned, R_etoile)
    return incertitude


