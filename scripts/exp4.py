import numpy as np

# Gegebene Werte
d_mm = [40.9, 58.8, 73.3, 87.5, 101.2, 115.3, 131.3, 153.1]
D_mm = 90  # Halber Durchmesser des Films
wavelength_nm = 1.54

# Umrechnung von mm zu m
d = np.array(d_mm) * 1e-3
D = D_mm * 1e-3

# Berechnung des Beugungswinkels
theta = np.arcsin(d / (2 * D))

# Berechnung der Gitterkonstante
a = (wavelength_nm * 1e-10) / (2 * np.sin(theta[0]))

print("Gitterkonstante a = {:.3f} nm".format(a))