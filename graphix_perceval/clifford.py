import numpy as np
import perceval.components as comp
import sympy as sp

# perceval representation of Clifford gates.
# see graphix.clifford module for the definitions and details of Clifford operatos for each index.
CLIFFORD_TO_PERCEVAL_BS = [
    [comp.BS(theta=0.0)],
    [comp.BS(theta=sp.pi, phi_bl=-sp.pi / 2, phi_br=-sp.pi / 2)],
    [comp.BS(theta=sp.pi, phi_tl=-sp.pi / 2, phi_bl=sp.pi / 2, phi_tr=sp.pi / 2, phi_br=sp.pi / 2)],
    [comp.BS(theta=0.0, phi_bl=sp.pi / 2, phi_br=sp.pi / 2)],
    [comp.BS(theta=0.0, phi_br=sp.pi / 2)],
    [comp.BS(theta=0.0, phi_br=-sp.pi / 2)],
    [comp.BS(theta=sp.pi / 2, phi_bl=3 * sp.pi / 2, phi_br=3 * sp.pi / 2)],
    [comp.BS(theta=sp.pi / 2, phi_tl=sp.pi / 2, phi_bl=-sp.pi / 2, phi_tr=-sp.pi / 2, phi_br=sp.pi / 2)],
    [comp.BS(theta=sp.pi / 2, phi_bl=sp.pi / 2, phi_br=3 * sp.pi / 2)],
    [comp.BS(theta=sp.pi, phi_tl=3 * sp.pi / 4, phi_tr=-3 * sp.pi / 4)],
    [comp.BS(theta=sp.pi, phi_tl=-3 * sp.pi / 4, phi_tr=3 * sp.pi / 4)],
    [comp.BS(theta=sp.pi / 2, phi_bl=sp.pi / 2, phi_br=sp.pi / 2)],
    [comp.BS(theta=sp.pi / 2, phi_tl=3 * sp.pi / 4, phi_bl=sp.pi / 4, phi_tr=sp.pi / 4, phi_br=3 * sp.pi / 4)],
    [comp.BS(theta=sp.pi / 2, phi_tr=sp.pi / 2, phi_br=3 * sp.pi / 2)],
    [comp.BS(theta=sp.pi / 2, phi_tl=sp.pi / 2, phi_bl=3 * sp.pi / 2)],
    [comp.BS(theta=sp.pi / 2, phi_tr=sp.pi, phi_br=3 * sp.pi)],
    [comp.BS(theta=sp.pi / 2, phi_bl=sp.pi, phi_tr=3 * sp.pi / 4, phi_br=sp.pi / 4)],
    [comp.BS(theta=sp.pi / 2, phi_tr=3 * sp.pi / 4, phi_br=5 * sp.pi / 4)],
    [comp.BS(theta=sp.pi / 2, phi_bl=sp.pi, phi_tr=sp.pi / 4, phi_br=3 * sp.pi / 4)],
    [comp.BS(theta=sp.pi / 2, phi_tr=5 * sp.pi / 4, phi_br=3 * sp.pi / 4)],
    [comp.BS(theta=sp.pi / 2, phi_tl=5 * sp.pi / 4, phi_bl=3 * sp.pi / 4)],
    [comp.BS(theta=sp.pi / 2, phi_tl=sp.pi / 2, phi_tr=sp.pi / 4, phi_br=5 * sp.pi / 4)],
    [comp.BS(theta=sp.pi / 2, phi_bl=sp.pi / 2, phi_tr=sp.pi / 4, phi_br=5 * sp.pi / 4)],
    [comp.BS(theta=sp.pi / 2, phi_tl=3 * sp.pi / 4, phi_bl=5 * sp.pi / 4)],
]

CLIFFORD_TO_PERCEVAL_POLAR = [
    [comp.WP(delta=0.0, xsi=0.0)],
    [comp.WP(delta=sp.pi / 2, xsi=sp.pi / 4), comp.PS(-sp.pi / 2)],
    [comp.WP(delta=sp.pi / 2, xsi=0.0), comp.WP(delta=sp.pi / 2, xsi=sp.pi / 4), comp.PS(-sp.pi / 2)],
    [comp.WP(delta=sp.pi / 2, xsi=0.0), comp.PS(-sp.pi / 2)],
    [comp.WP(delta=-sp.pi / 4, xsi=0.0), comp.PS(sp.pi / 4)],
    [comp.WP(delta=sp.pi / 4, xsi=0.0), comp.PS(7 * sp.pi / 4)],
    [comp.WP(delta=sp.pi / 2, xsi=np.pi / 8), comp.PS(3 * sp.pi / 2)],
    [comp.WP(delta=3 * sp.pi / 4, xsi=np.pi / 4), comp.PS(sp.pi)],
    [comp.WP(delta=sp.pi / 2, xsi=sp.pi / 8), comp.WP(delta=sp.pi / 2, xsi=sp.pi / 4), comp.PS(sp.pi)],
    [comp.WP(delta=-sp.pi / 4, xsi=sp.pi), comp.WP(delta=sp.pi / 2, xsi=sp.pi / 4), comp.PS(sp.pi)],
    [comp.WP(delta=sp.pi / 4, xsi=sp.pi), comp.WP(delta=sp.pi / 2, xsi=sp.pi / 4), comp.PS(-sp.pi)],
    [comp.WP(delta=sp.pi / 2, xsi=3 * sp.pi / 8), comp.PS(sp.pi / 2)],
    [comp.WP(delta=sp.pi / 2, xsi=3 * sp.pi / 8), comp.WP(delta=sp.pi / 2, xsi=sp.pi / 4)],
    [comp.WP(delta=sp.pi / 4, xsi=sp.pi / 4), comp.WP(delta=sp.pi / 2, xsi=0.0)],
    [comp.WP(delta=sp.pi / 4, xsi=3 * sp.pi / 4), comp.WP(delta=sp.pi / 2, xsi=0.0)],
    [comp.WP(delta=sp.pi / 4, xsi=sp.pi / 4), comp.PS(sp.pi)],
    [comp.WP(delta=sp.pi / 4, xsi=0.0), comp.WP(delta=sp.pi / 2, xsi=sp.pi / 8)],
    [comp.WP(delta=sp.pi / 4, xsi=0.0), comp.WP(delta=sp.pi / 2, xsi=3 * sp.pi / 8), comp.PS(sp.pi)],
    [comp.WP(delta=sp.pi / 4, xsi=sp.pi / 2), comp.WP(delta=sp.pi / 2, xsi=3 * sp.pi / 8), comp.PS(sp.pi)],
    [comp.WP(delta=sp.pi / 4, xsi=sp.pi / 2), comp.WP(delta=sp.pi / 2, xsi=5 * sp.pi / 8)],
    [comp.WP(delta=sp.pi / 4, xsi=0.0), comp.WP(delta=sp.pi / 4, xsi=sp.pi / 4), comp.PS(sp.pi)],
    [comp.WP(delta=sp.pi / 4, xsi=sp.pi / 4), comp.WP(delta=sp.pi / 2, xsi=sp.pi / 8)],
    [comp.WP(delta=sp.pi / 4, xsi=0.0), comp.WP(delta=sp.pi / 4, xsi=3 * sp.pi / 4)],
    [comp.WP(delta=sp.pi / 4, xsi=sp.pi / 2), comp.WP(delta=sp.pi / 4, xsi=sp.pi / 4), comp.PS(sp.pi)],
]
