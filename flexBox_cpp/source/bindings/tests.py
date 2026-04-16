import sys
import os


sys.path.append(os.path.abspath("../build/bindings"))



import numpy as np
import flexBoxPy

# =========================
# 1. Parameter
# =========================
params = {
    "maxIt": 50,
    "verbose": 1,
    "tol": 1e-4
}

# =========================
# 2. Primal Variablen
# =========================
# Beispiel: 1D Signal mit 5 Pixeln
x0 = np.zeros(5)

x = [x0]  # Liste von Primal-Variablen
dims = [[5]]  # Dimensionen (muss zur Größe passen)

# =========================
# 3. Dual Terms definieren
# =========================

dual_term = {
    "type": "L2",          # Prox-Typ
    "factor": 1.0,         # alpha
    "operator": [
        {
            "type": "identity",
            "nPx": 5
        }
    ],
    "f": [
        np.ones(5)  # Ziel-Daten (z.B. Messdaten)
    ]
}

duals = [dual_term]

# =========================
# 4. Mapping Dual → Primal
# =========================
# "DcP": welcher Dual-Term gehört zu welchem Primal
DcP = [
    [1]   # Achtung: 1-basiert (wird im C++ zu 0-basiert gemacht)
]

# =========================
# 5. Input Dictionary
# =========================
input_data = {
    "firstRun": True,
    "params": params,
    "x": x,
    "dims": dims,
    "duals": duals,
    "DcP": DcP
}

# =========================
# 6. Algorithmus starten
# =========================
result = flexBoxPy.run(input_data)

# =========================
# 7. Ergebnisse anzeigen
# =========================
print("Primal Ergebnis:")
for p in result["primal"]:
    print(np.array(p))

print("\nDual Ergebnis:")
for d in result["dual"]:
    print(np.array(d))
