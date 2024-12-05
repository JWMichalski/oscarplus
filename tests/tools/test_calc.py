import numpy as np
import numpy.testing as npt
from oscarplus.tools import calc


def test_median_angle_odd():
    """tests the median angle for an odd number of angles"""
    angles_odd = np.array([355, 5, 0, 10, 0, 180, 0, 180, 180])
    assert calc.median_angle(angles_odd) == 5.0


def test_median_angle_even():
    """tests the median angle for an even number of angles with NaNs"""
    angles_even = np.array([355, 0, 10, 20, 30, 40, np.nan, np.nan])
    assert calc.median_angle(angles_even) == 15.0


def test_median_angle_even_large_angle_difference():
    """tests the case where the difference between the two angles is larger than 180"""
    assert calc.median_angle(np.array([0, 350])) == 355.0


def test_angle_average_large_difference():
    """tests the case where the difference between the two angles is larger than 180"""
    assert calc.angle_average(0, 350) == 355.0


def test_angle_average_small_difference():
    """tests the case where the difference between the two angles is smaller than 180"""
    assert calc.angle_average(0, 10) == 5.0


def test_angle_average_angles_large_average():
    """tests the case where the average of the two angles is larger than 180"""
    assert calc.angle_average(350, 20) == 5.0


def test_rotate_vectors():
    """tests the case where the vectors are rotated by 90 degrees"""
    U = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    V = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
    U_rot, V_rot = calc.rotate_vectors(U, V, 45)
    npt.assert_array_almost_equal(
        U_rot,
        np.array(
            [[1.41421356, 0.0, 0.0], [0.0, 0.70710678, 0.0], [0.0, 0.0, 0.70710678]]
        ),
    )
    npt.assert_array_almost_equal(
        V_rot,
        np.array([[0.0, 0.0, 0.0], [0.0, -0.70710678, 0.0], [0.0, 0.0, 0.70710678]]),
    )


def test_divergence_uniform():
    """tests the divergence of a f=(x,y) vector field"""
    f_x = np.fromfunction(lambda x, y: x, (10, 10))
    f_y = np.fromfunction(lambda x, y: y, (10, 10))
    div_expected = np.full((10, 10), 2)

    div = calc.divergence([f_x, f_y])

    npt.assert_array_almost_equal(div, div_expected)


def test_divergence():
    """tests the divergence of a f=(x^2,y) vector field"""
    f_x = np.fromfunction(lambda x, y: (x) ** 2, (10, 10))
    f_y = np.fromfunction(lambda x, y: (y), (10, 10))
    div_expected = np.fromfunction(lambda x, y: 2 * (x) + 1, (10, 10))

    div = calc.divergence([f_x, f_y])

    # ignore edges as gradient will not be correct there:
    npt.assert_array_almost_equal(div[1:-1, 1:-1], div_expected[1:-1, 1:-1])


def test_divergence_small_values():
    """tests the divergence of a f=(x/100^2,y/100) vector field"""
    f_x = np.fromfunction(lambda x, y: (x / 100) ** 2, (100, 100))
    f_y = np.fromfunction(lambda x, y: (y / 100), (100, 100))
    div_expected = np.fromfunction(lambda x, y: (2 * (x / 10000) + 1 / 100), (100, 100))

    div = calc.divergence([f_x, f_y])

    # ignore edges as gradient will not be correct there:
    npt.assert_array_almost_equal(div[1:-1, 1:-1], div_expected[1:-1, 1:-1])


def test_curl_uniform():
    """tests the curl of a f=(-y,x) vector field"""
    f_x = np.fromfunction(lambda x, y: -y, (10, 10))
    f_y = np.fromfunction(lambda x, y: x, (10, 10))
    curl_expected = np.full((10, 10), 2)

    curl = calc.curl([f_x, f_y])

    npt.assert_array_almost_equal(curl, curl_expected)


def test_curl():
    """tests the curl of a f=(x*y,x) vector field"""
    f_x = np.fromfunction(lambda x, y: x * y, (10, 10))
    f_y = np.fromfunction(lambda x, y: x, (10, 10))
    curl_expected = np.fromfunction(lambda x, y: 1 - x, (10, 10))

    curl = calc.curl([f_x, f_y])

    npt.assert_array_almost_equal(curl, curl_expected)


def test_curl_small_values():
    """tests the curl of a f=(x/100*y/100,x/100) vector field"""
    f_x = np.fromfunction(lambda x, y: x / 100 * y / 100, (10, 10))
    f_y = np.fromfunction(lambda x, y: x / 100, (10, 10))
    curl_expected = np.fromfunction(lambda x, y: 1 / 100 - x / 10000, (10, 10))

    curl = calc.curl([f_x, f_y])

    npt.assert_array_almost_equal(curl, curl_expected)


def test_shear_rate_uniform():
    """tests the shear rate of a f=(y,x) vector field"""
    f_x = np.fromfunction(lambda x, y: y, (10, 10))
    f_y = np.fromfunction(lambda x, y: x, (10, 10))
    shear_expected = np.full((10, 10), 2)

    shear = calc.shear_rate([f_x, f_y])

    npt.assert_array_almost_equal(shear, shear_expected)


def test_shear_rate():
    """tests the shear rate of a f=(y^2,x) vector field"""
    f_x = np.fromfunction(lambda x, y: (y) ** 2, (10, 10))
    f_y = np.fromfunction(lambda x, y: (x), (10, 10))
    shear_expected = np.fromfunction(lambda x, y: 2 * (y) + 1, (10, 10))

    shear = calc.shear_rate([f_x, f_y])

    # ignore edges as gradient will not be correct there:
    npt.assert_array_almost_equal(shear[1:-1, 1:-1], shear_expected[1:-1, 1:-1])


def test_strain_rate_symmetric():
    """tests the strain rate of a f=(y,x) vector field"""
    f_x = np.fromfunction(lambda x, y: x * y, (10, 10))
    f_y = np.fromfunction(lambda x, y: x * y, (10, 10))
    strain_expected = np.fromfunction(lambda x, y: np.sqrt(2 * (x**2 + y**2)), (10, 10))

    strain = calc.strain_rate([f_x, f_y])

    npt.assert_array_almost_equal(strain, strain_expected)


def test_strain_rate():
    """tests the strain rate of a f=(y,x) vector field"""
    f_x = np.fromfunction(lambda x, y: (x**2), (10, 10))
    f_y = np.fromfunction(lambda x, y: (y**2), (10, 10))
    strain_expected = np.fromfunction(
        lambda x, y: np.sqrt((2 * x - 2 * y) ** 2), (10, 10)
    )

    strain = calc.strain_rate([f_x, f_y])

    # ignore edges as gradient may not be correct there:
    npt.assert_array_almost_equal(strain[1:-1, 1:-1], strain_expected[1:-1, 1:-1])


def test_kinetic_energy():
    """tests the kinetic energy of a f=(y,x) vector field"""
    f_x = np.full((10, 10), 2)
    f_y = np.full((10, 10), 4)
    kinetic_expected = np.full((10, 10), 10)

    kinetic = calc.kinetic_energy_density([f_x, f_y])

    npt.assert_array_almost_equal(kinetic, kinetic_expected)


def test_enstrophy_uniform():
    """tests the enstrophy of a f=(y,x) vector field"""
    f_x = np.fromfunction(lambda x, y: x, (10, 10))
    f_y = np.fromfunction(lambda x, y: y, (10, 10))
    enstrophy_expected = np.full((10, 10), 0)

    enstrophy = calc.enstrophy([f_x, f_y])

    npt.assert_array_almost_equal(enstrophy, enstrophy_expected)

    """tests the enstrophy of a f=(x^2,y) vector field"""
    f_x = np.fromfunction(lambda x, y: (x) ** 2, (10, 10))
    f_y = np.fromfunction(lambda x, y: x * y, (10, 10))
    enstrophy_expected = np.fromfunction(lambda x, y: (2 * x - x) ** 2, (10, 10))

    enstrophy = calc.enstrophy([f_x, f_y])

    npt.assert_array_almost_equal(enstrophy[1:-1, 1:-1], enstrophy_expected[1:-1, 1:-1])
