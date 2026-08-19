import math

import numpy as np
import pytest

import SolarSim as sim


def test_gravity_points_toward_other_body():
    positions = np.array([[0.0, 0.0], [4.0e9, 0.0]])
    masses = np.array([2.0e20, 5.0e20])

    acceleration = sim.accel_majors_numpy(positions, masses)

    assert acceleration[0, 0] > 0.0
    assert acceleration[1, 0] < 0.0
    np.testing.assert_allclose(acceleration[:, 1], 0.0, atol=0.0)


def test_internal_gravitational_forces_are_equal_and_opposite():
    positions = np.array([[-3.0e9, 2.0e9], [5.0e9, -1.0e9]])
    masses = np.array([3.0e20, 7.0e20])

    acceleration = sim.accel_majors_numpy(positions, masses)
    forces = masses[:, None] * acceleration

    np.testing.assert_allclose(forces[0], -forces[1], rtol=1e-14, atol=1e-12)


def test_major_body_initial_state_is_barycentric():
    _, masses, positions, velocities = sim.make_major_body_state()

    center_of_mass = (masses[:, None] * positions).sum(axis=0) / masses.sum()
    total_momentum = (masses[:, None] * velocities).sum(axis=0)

    np.testing.assert_allclose(center_of_mass, 0.0, atol=1e-6)
    np.testing.assert_allclose(total_momentum, 0.0, atol=2e16)


@pytest.mark.skipif(not sim.NUMBA_AVAILABLE, reason="Numba is not installed")
def test_numpy_and_numba_one_step_agree():
    _, masses, positions, velocities = sim.make_major_body_state()
    asteroid_positions, asteroid_velocities = sim.make_asteroids(
        8, positions[0], velocities[0], masses[0], seed=7
    )
    dt = 3600.0

    numpy_result = sim.step_numpy(
        positions.copy(),
        velocities.copy(),
        asteroid_positions.copy(),
        asteroid_velocities.copy(),
        masses,
        dt,
    )
    numba_result = sim.step_numba(
        positions.copy(),
        velocities.copy(),
        asteroid_positions.copy(),
        asteroid_velocities.copy(),
        masses,
        dt,
        sim.G,
        sim.SOFTENING,
    )

    for numpy_array, numba_array in zip(numpy_result, numba_result):
        np.testing.assert_allclose(numpy_array, numba_array, rtol=1e-12, atol=1e-7)


def test_simple_two_body_orbit_stays_bounded_for_one_year():
    sun_mass = 1.9885e30
    earth_mass = 5.97237e24
    masses = np.array([sun_mass, earth_mass])
    total_mass = masses.sum()

    relative_position = np.array([sim.AU, 0.0])
    relative_speed = math.sqrt(sim.G * total_mass / sim.AU)
    positions = np.array(
        [
            -earth_mass / total_mass * relative_position,
            sun_mass / total_mass * relative_position,
        ]
    )
    velocities = np.array(
        [
            [0.0, -earth_mass / total_mass * relative_speed],
            [0.0, sun_mass / total_mass * relative_speed],
        ]
    )
    asteroid_positions = np.zeros((0, 2))
    asteroid_velocities = np.zeros((0, 2))

    dt = 6.0 * 3600.0
    steps = round(sim.SECONDS_PER_YEAR / dt)
    radii = []

    for _ in range(steps):
        positions, velocities, asteroid_positions, asteroid_velocities = sim.step_numpy(
            positions,
            velocities,
            asteroid_positions,
            asteroid_velocities,
            masses,
            dt,
        )
        radii.append(np.linalg.norm(positions[1] - positions[0]))

    radii = np.asarray(radii)
    assert np.isfinite(radii).all()
    assert radii.min() > 0.995 * sim.AU
    assert radii.max() < 1.005 * sim.AU


def test_asteroid_generation_is_deterministic_with_seed():
    sun_position = np.array([1.0e9, -2.0e9])
    sun_velocity = np.array([10.0, -20.0])

    first = sim.make_asteroids(12, sun_position, sun_velocity, 1.9885e30, seed=42)
    second = sim.make_asteroids(12, sun_position, sun_velocity, 1.9885e30, seed=42)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_asteroids_are_initialized_relative_to_sun_state():
    count = 10
    sun_position = np.array([3.0 * sim.AU, -2.0 * sim.AU])
    sun_velocity = np.array([120.0, -240.0])

    origin_positions, origin_velocities = sim.make_asteroids(
        count, np.zeros(2), np.zeros(2), 1.9885e30, seed=123
    )
    shifted_positions, shifted_velocities = sim.make_asteroids(
        count, sun_position, sun_velocity, 1.9885e30, seed=123
    )

    expected_position_shift = np.broadcast_to(sun_position, (count, 2))
    expected_velocity_shift = np.broadcast_to(sun_velocity, (count, 2))
    np.testing.assert_allclose(shifted_positions - origin_positions, expected_position_shift)
    np.testing.assert_allclose(shifted_velocities - origin_velocities, expected_velocity_shift)

    distances_from_sun = np.linalg.norm(shifted_positions - sun_position, axis=1)
    assert np.all(distances_from_sun >= 2.1 * sim.AU)
    assert np.all(distances_from_sun <= 3.3 * sim.AU)
