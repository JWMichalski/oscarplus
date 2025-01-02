"""
Numerical tools for calculations
================================
Functions for numerical calculations that do not depend on the OSCAR dataset format

Functions
---------
- angle_average :
    compute the average of two angles
- rotate_vectors :
    rotate vectors with components U and V by rotation_angle degrees counterclockwise
- divergence :
    computes the divergence of the vector field
- curl :
    calculate the curl of a vector field
- shear_rate :
    calculate the shear rate of a vector field
- strain_rate :
    calculate the strain rate of a vector field
- kinetic_energy_density :
    calculate the kinetic energy of a vector field
- enstrophy :
    calculate the enstrophy of a vector field
- wrap_numpy_2D_vector_calc :
    wrapper function for xarray to perform calculations on 2D vector fields
- median_angle :
    find the median angle of a set of angles in degrees
"""

import numpy as np
import xarray as xr


def angle_average(angle1, angle2):
    """
    Compute the average of two angles

    Parameters
    ----------
    angle1 : ``float``
        First angle in degrees in range [0, 360].
    angle2 : ``float``
        Second angle in degrees in range [0, 360].
    Returns
    -------
    ``float``
        Average of the two angles in degrees.
    """
    if np.isnan(angle1) and np.isnan(angle2):
        return np.nan
    if np.isnan(angle1):
        return angle2
    if np.isnan(angle2):
        return angle1
    if angle1 < 0 or angle1 >= 360:
        raise ValueError("angle1 must be in range [0, 360)")
    if angle2 < 0 or angle2 >= 360:
        raise ValueError("angle2 must be in range [0, 360)")

    if abs(angle1 - angle2) <= 180:
        return (angle1 + angle2) / 2
    else:
        average = (angle1 + angle2) / 2 + 180
        if average >= 360:
            average -= 360
        return average


def rotate_vectors(U, V, rotation_angle):
    """
    Rotate vectors with components U and V by rotation_angle degrees counterclockwise

    Parameters
    ----------
    U : ``numpy.ndarray``
        Array of any shape with the x-components of the vectors.
    V : ``numpy.ndarray``
        Array of the same shape as U with the y-components of the vectors.
    rotation_angle : ``float``
        Angle in degrees by which to rotate the vectors counterclockwise.
    Returns
    -------
    U_rot : ``numpy.ndarray``
        Array of the same shape as U with the x-components of the rotated vectors.
    V_rot : ``numpy.ndarray``
        Array of the same shape as V with the y-components of the rotated vectors.
    """
    if not isinstance(U, np.ndarray) or not isinstance(V, np.ndarray):
        raise TypeError("U and V must be numpy arrays.")
    if U.shape != V.shape:
        raise ValueError("U and V must have the same shape.")

    rotation_angle = np.radians(rotation_angle)
    rotation_matrix = np.array(
        [
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)],
        ]
    )
    vectors = np.stack([U, V])
    vectors_rot = np.apply_along_axis(
        arr=vectors,
        axis=0,
        func1d=lambda vector, matrix: vector.dot(matrix),
        matrix=rotation_matrix,
    )
    assert (
        vectors_rot.shape == vectors.shape
    ), "Matrix must have the shame shape before and after rotation."
    U_rot, V_rot = np.split(vectors_rot, 2)
    U_rot = U_rot.squeeze()
    V_rot = V_rot.squeeze()
    return U_rot, V_rot


def divergence(f):
    """
    Computes the divergence of a vector field f

    Parameters
    ----------
    f : ``List of np.ndarray``
        List where every item of the list is one dimension of the vector field.
    Returns
    -------
    ``np.ndarray``
        Single np.ndarray of the same shape as each of the items in f,
        which corresponds to a scalar field.
    """
    num_dims = len(f)
    shapes = [arr.shape for arr in f]
    if len(set(shapes)) != 1:
        raise ValueError("All arrays in f must have the same shape.")
    return np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(num_dims)])


def curl(f):
    """
    Computes the curl of a vector field f

    Parameters
    ----------
    f : ``list of np.ndarray``
        A list of two arrays, each representing a component of the vector field.
    Returns
    -------
    ``np.ndarray``
        An array representing the curl of the vector field.
    """
    if len(f) != 2:
        raise ValueError("f must have exactly two components.")
    if f[0].shape != f[1].shape:
        raise ValueError("f[0] and f[1] must have the same shape.")
    dFx_dy = np.gradient(f[0], axis=1)
    dFy_dx = np.gradient(f[1], axis=0)
    return dFy_dx - dFx_dy


def shear_rate(f):
    """
    Computes the shear rate of a vector field f

    Parameters
    ----------
    f : ``list of np.ndarray``
        A list of two arrays, each representing a component of the vector field.
    Returns
    -------
    ``np.ndarray``
        An array representing the shear rate of the vector field.
    """
    if len(f) != 2:
        raise ValueError("f must have exactly two components.")
    if f[0].shape != f[1].shape:
        raise ValueError("f[0] and f[1] must have the same shape.")
    dFx_dy = np.gradient(f[0], axis=1)
    dFy_dx = np.gradient(f[1], axis=0)
    return dFy_dx + dFx_dy


def strain_rate(f):
    """
    Computes the strain rate of a vector field f

    Parameters
    ----------
    f : ``list of np.ndarray``
        A list of two arrays, each representing a component of the vector field.
    Returns
    -------
    ``np.ndarray``
        An array representing the strain rate of the vector field.
    """
    if len(f) != 2:
        raise ValueError("f must have exactly two components.")
    if f[0].shape != f[1].shape:
        raise ValueError("f[0] and f[1] must have the same shape.")
    dFx_dx = np.gradient(f[0], axis=0)
    dFy_dy = np.gradient(f[1], axis=1)
    dFx_dy = np.gradient(f[0], axis=1)
    dFy_dx = np.gradient(f[1], axis=0)
    return np.sqrt((dFx_dx - dFy_dy) ** 2 + (dFy_dx + dFx_dy) ** 2)


def kinetic_energy_density(f):
    """
    Computes the kinetic energy of a vector field f

    Parameters
    ----------
    f : ``list of np.ndarray``
        A list of two arrays, each representing a component of the vector field.
    Returns
    -------
    ``np.ndarray``
        An array representing the kinetic energy of the vector field.
    """
    if len(f) != 2:
        raise ValueError("f must have exactly two components.")
    if f[0].shape != f[1].shape:
        raise ValueError("f[0] and f[1] must have the same shape.")

    out = 0.5 * (f[0] ** 2 + f[1] ** 2)

    assert out.shape == f[0].shape, "Output shape must be the same as the input shape."
    return out


def enstrophy(f):
    """
    Computes the enstrophy of a vector field f

    Parameters
    ----------
    f : ``list of np.ndarray``
        A list of two arrays, each representing a component of the vector field.
    Returns
    -------
    ``np.ndarray``
        An array representing the enstrophy of the vector field.
    """
    if len(f) != 2:
        raise ValueError("f must have exactly two components.")
    if f[0].shape != f[1].shape:
        raise ValueError("f[0] and f[1] must have the same shape.")
    dFx_dx = np.gradient(f[0], axis=0)
    dFy_dy = np.gradient(f[1], axis=1)
    return (dFy_dy - dFx_dx) ** 2


def wrap_numpy_2D_vector_calc(DA_u, DA_v, selected_calculation):
    """
    Wrapper function for xarray to perform calculations on 2D vector fields
    using np.array based functions

    Parameters
    ----------
    DA_u : ``xarray.DataArray``
        xarray.DataArray, containing the u-component of the vector field.
    DA_v : ``xarray.DataArray``
        xarray.DataArray, containing the v-component of the vector field.
        Must have the same shape as DA_u.
    selected_calculation : ``str``
        String, containing the name of the calculation to be performed.
        Supported calculations are: 'Divergence', 'Curl', 'ShearRate',
        'StrainRate', 'KineticEnergyDensity', 'Enstrophy'.
    Returns
    -------
    ``xarray.DataArray``
        xarray.DataArray, containing the result of the calculation.
    """
    if selected_calculation == "Divergence":
        div = divergence([DA_v.values, DA_u.values])
        output_da = xr.DataArray(div, coords=DA_u.coords)
    elif selected_calculation == "Curl":
        cur = -curl([DA_v.values, DA_u.values])
        output_da = xr.DataArray(cur, coords=DA_u.coords)
    elif selected_calculation == "ShearRate":
        sh = -shear_rate([DA_v.values, DA_u.values])
        output_da = xr.DataArray(sh, coords=DA_u.coords)
    elif selected_calculation == "StrainRate":
        sr = strain_rate([DA_v.values, DA_u.values])
        output_da = xr.DataArray(sr, coords=DA_u.coords)
    elif selected_calculation == "KineticEnergyDensity":
        ked = kinetic_energy_density([DA_v.values, DA_u.values])
        output_da = xr.DataArray(ked, coords=DA_u.coords)
    elif selected_calculation == "Enstrophy":
        ens = enstrophy([DA_v.values, DA_u.values])
        output_da = xr.DataArray(ens, coords=DA_u.coords)
    else:
        raise ValueError("The selected calculation is not supported")
    return output_da


def median_angle(angles):
    """
    Find the median angle of a set of angles in degrees.

    Median of angles is defined as the angle that minimizes
    the sum of the absolute differences.

    Parameters
    ----------
    angles : ``numpy.array``
        Array of 9 angles to find median of in degrees.
    Returns
    -------
    median_angle : ``float``
        Median angle in degrees.
    Raises
    ------
    ValueError
        If the input array does not have 9 elements
    """
    angles = angles[~np.isnan(angles)]
    if angles.ndim > 1:  # flatten the array if it is not 1D
        angles = angles.flatten()
    if angles.size == 1:  # return the only angle if only one
        return angles[0]
    if angles.size == 0:  # return NaN if empty
        return np.nan

    assert not np.any(np.isnan(angles)), "Input array must not contain NaNs"
    assert angles.ndim == 1, "Input array must be 1D"
    assert angles.size > 1, "Input array must have at least 2 elements"

    angles = np.sort(angles)
    # calculate the difference between the angles
    # diff[n] = absolute distance between diff[n] and diff[n+1] with wrap around
    diff = np.diff(angles, append=angles[0])
    diff = np.where(diff < 0, 360 + diff, diff)
    diff = np.where(diff > 180, 360 - diff, diff)
    largest_diff = np.argmax(diff)  # find the position of the largest difference
    # determine the median angle
    if angles.size % 2 == 1:
        median_angle = angles[largest_diff - np.int_(angles.size / 2)]
    else:
        median_angle = angle_average(
            angles[largest_diff - np.int_(angles.size / 2)],
            angles[largest_diff - np.int_(angles.size / 2) + 1],
        )
    assert (
        0 <= median_angle < 360
    ), f"median angle is {median_angle}, outside of [0, 360) range."
    return median_angle
