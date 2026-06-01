"""Coordinate transforms between Y-Z (detector) and R-Eta (polar) spaces.

Source: SIMR-xrd/SIMR_xrd/utils/coord_transform.py (most complete implementation)

Terminology:
    Y -> horizontal direction, Z -> vertical direction in detector frame
    R -> radial distance from beam center (pixels)
    Eta -> angular coordinate on a diffraction ring (degrees)

Eta convention in detector frame:
    Sweep from TOP to RIGHT to BOTTOM (clockwise): ETA varies from +180 to +90 to +0
    Sweep from TOP to LEFT to BOTTOM (counter-clockwise): ETA varies from -180 to -90 to -0
"""

import numpy as np


# ----------------------------
def YZ_from_REta(R_inPx, Eta_inDeg, Ypx_BC, Zpx_BC):
    """Calculate Y-Z pixel coordinate of a point/array of points from R (in pixels) and Eta (in degrees) coordinates
    Args:
        R_inPx (float or arr): R values in pixels
        Eta_inDeg (float or arr): Eta values in degrees
        Ypx_BC (float): Beam Center Y pixel coordinate
        Zpx_BC (float): Beam Center Z pixel coordinate
    Returns:
        Ypx (int or arr): Y pixel coordinate (col index)
        Zpx (int or arr): Z pixel coordinate (row index)
    """

    Ypx = Ypx_BC + (R_inPx * np.sin(np.deg2rad(Eta_inDeg)))
    Zpx = Zpx_BC + (R_inPx * np.cos(np.deg2rad(Eta_inDeg)))

    return (Ypx, Zpx)


# ----------------------------
def REta_from_YZ(Ypx, Zpx, Ypx_BC, Zpx_BC):
    """Calculate R-Eta coordinates of a point/array of points from Y-Z pixel coordinates
    Args:
        Ypx (int or arr): Y pixel coordinate (col index)
        Zpx (int or arr): Z pixel coordinate (row index)
        Ypx_BC (float): Beam Center Y pixel coordinate
        Zpx_BC (float): Beam Center Z pixel coordinate
    Return:
        R_inPx (float or arr): R values in pixels
        Eta_inDeg (float or arr): Eta values in degrees
    """

    R_inPx = ((Ypx_BC - Ypx)**2 + (Zpx_BC - Zpx)**2)**0.5
    Eta_inDeg = np.rad2deg(np.arccos((Zpx - Zpx_BC) / R_inPx))

    # Convert Eta_inDeg to -ve where Ypx < Ypx_BC
    Eta_inDeg = Eta_inDeg * ((Ypx - Ypx_BC) / np.abs(Ypx - Ypx_BC))

    return (R_inPx, Eta_inDeg)


# ----------------------------
def beamcenter_from_YZREta(Ypx, Zpx, R_inPx, Eta_inDeg):
    """Calculate Beam Center Pixel coordinates from known Y, Z, R and Eta values for a single point or array of points.
    Args:
        Ypx (int or arr): Y pixel coordinate (col index)
        Zpx (int or arr): Z pixel coordinate (row index)
        R_inPx (float or arr): R values in pixels
        Eta_inDeg (float or arr): Eta values in degrees
    Return:
        Ypx_BC (float or arr): Beam Center Y pixel coordinate
        Zpx_BC (float or arr): Beam Center Z pixel coordinate
    """

    Ypx_BC = Ypx - (R_inPx * np.sin(np.deg2rad(Eta_inDeg)))
    Zpx_BC = Zpx - (R_inPx * np.cos(np.deg2rad(Eta_inDeg)))

    return (Ypx_BC, Zpx_BC)
