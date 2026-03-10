import numpy as np
import pytest
from numpy.testing import assert_allclose
from etas.simulation import *


# ── inv_time_cdf_approx ───────────────────────────────────────────────────────

class TestInvTimeCdfApprox:
    """Inverse CDF should map [0,1] → non-negative times and be monotone."""

    def setup_method(self):
        self.c = 0.01
        self.tau = 10.0
        self.omega = 0.5

    def test_output_nonnegative(self):
        p = np.linspace(0.01, 0.99, 100)
        t = inv_time_cdf_approx(p, self.c, self.tau, self.omega)
        assert np.all(t >= 0)

    def test_monotone_increasing(self):
        p = np.linspace(0.01, 0.99, 200)
        t = inv_time_cdf_approx(p, self.c, self.tau, self.omega)
        assert np.all(np.diff(t) > 0)

    def test_scalar_input(self):
        t = inv_time_cdf_approx(0.5, self.c, self.tau, self.omega)
        assert np.isfinite(t)

    def test_boundary_low(self):
        t = inv_time_cdf_approx(1e-6, self.c, self.tau, self.omega)
        assert t >= 0 and np.isfinite(t)

    def test_boundary_high(self):
        t = inv_time_cdf_approx(1 - 1e-6, self.c, self.tau, self.omega)
        assert t >= 0 and np.isfinite(t)

    def test_piecewise_branch_selection(self):
        """Values below and above tau should use different branches."""
        p_low = np.array([1e-4])   # p < tau → power-law branch
        p_high = np.array([0.99])  # p > tau (tau=10) → exponential branch
        t_low = inv_time_cdf_approx(p_low, self.c, self.tau, self.omega)
        t_high = inv_time_cdf_approx(p_high, self.c, self.tau, self.omega)
        assert t_low < t_high


# ── simulate_aftershock_time_approx ──────────────────────────────────────────

class TestSimulateAftershockTimeApprox:

    def test_output_shape(self):
        t = simulate_aftershock_time_approx(1, 0.5, 1, size=500)
        assert t.shape == (500,)

    def test_nonnegative(self):
        t = simulate_aftershock_time_approx(1, 0.5, 1, size=500)
        assert np.all(t >= 0)

    def test_default_size(self):
        t = simulate_aftershock_time_approx(1, 0.5, 1)
        assert np.isscalar(t) or t.shape == (1,)

    def test_finite(self):
        t = simulate_aftershock_time_approx(1, 0.5, 1, size=200)
        assert np.all(np.isfinite(t))

    def test_median_order_of_magnitude(self):
        """Median time should be in a physically plausible range (days)."""
        t = simulate_aftershock_time_approx(1, 0.5, 1, size=5000)
        assert 0 < np.median(t) < 1e4


# ── simulate_aftershock_time ──────────────────────────────────────────────────

class TestSimulateAftershockTime:

    def test_output_shape(self):
        t = simulate_aftershock_time(1, 0.5, 1, size=500)
        assert t.shape == (500,)

    def test_nonnegative(self):
        t = simulate_aftershock_time(1, 0.5, 1, size=500)
        assert np.all(t >= 0)

    def test_finite(self):
        t = simulate_aftershock_time(1, 0.5, 1, size=500)
        assert np.all(np.isfinite(t))

    def test_median_order_of_magnitude(self):
        t = simulate_aftershock_time(1, 0.5, 1, size=5000)
        assert 0 < np.median(t) < 1e4

    def test_approx_and_exact_similar_median(self):
        """Exact and approximate samplers should agree within an order of magnitude."""
        np.random.seed(42)
        t_exact = simulate_aftershock_time(1, 0.5, 1, size=5000)
        np.random.seed(42)
        t_approx = simulate_aftershock_time_approx(1, 0.5, 1, size=5000)
        assert abs(np.log10(np.median(t_exact)) - np.log10(np.median(t_approx))) < 1


# ── simulate_aftershock_time_untapered ───────────────────────────────────────

class TestSimulateAftershockTimeUntapered:

    def test_output_shape(self):
        t = simulate_aftershock_time_untapered(1, 0.5, size=500)
        assert t.shape == (500,)

    def test_nonnegative(self):
        t = simulate_aftershock_time_untapered(1, 0.5, size=500)
        assert np.all(t >= 0)

    def test_finite(self):
        t = simulate_aftershock_time_untapered(1, 0.5, size=500)
        assert np.all(np.isfinite(t))

    def test_closed_form_correctness(self):
        """
        For untapered power law, P(T > t) = ((t+c)/c)^(-omega).
        Check empirical CCDF matches at a specific quantile.
        """
        np.random.seed(0)
        c = 0.01
        omega = 0.5
        t = simulate_aftershock_time_untapered(np.log10(c), omega, size=50000)
        t_check = 1.0  # 1 day
        empirical_ccdf = np.mean(t > t_check)
        theoretical_ccdf = np.power((t_check + c) / c, -omega)
        assert_allclose(empirical_ccdf, theoretical_ccdf, rtol=0.05)


# ── simulate_aftershock_place ─────────────────────────────────────────────────

class TestSimulateAftershockPlace:

    def setup_method(self):
        self.mi = np.full(1000, 4.0)
        self.mc = 3.0

    def test_output_shape(self):
        x, y = simulate_aftershock_place(1, 0.5, 1.5, self.mi, self.mc)
        assert x.shape == (1000,) and y.shape == (1000,)

    def test_finite(self):
        x, y = simulate_aftershock_place(1, 0.5, 1.5, self.mi, self.mc)
        assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))

    def test_azimuth_symmetry(self):
        """x and y should have near-zero mean (isotropic distribution)."""
        x, y = simulate_aftershock_place(1, 0.5, 1.5, self.mi, self.mc)
        assert abs(np.mean(x)) < 5
        assert abs(np.mean(y)) < 5

    def test_larger_magnitude_larger_radius(self):
        """Higher magnitude events should produce larger median offsets."""
        mi_small = np.full(2000, 3.0)
        mi_large = np.full(2000, 6.0)
        x_s, y_s = simulate_aftershock_place(1, 0.5, 1.5, mi_small, self.mc)
        x_l, y_l = simulate_aftershock_place(1, 0.5, 1.5, mi_large, self.mc)
        r_small = np.sqrt(x_s**2 + y_s**2)
        r_large = np.sqrt(x_l**2 + y_l**2)
        assert np.median(r_large) > np.median(r_small)


# ── simulate_aftershock_radius ────────────────────────────────────────────────

class TestSimulateAftershockRadius:

    def setup_method(self):
        self.mi = np.full(1000, 4.0)
        self.mc = 3.0

    def test_output_shape(self):
        r = simulate_aftershock_radius(1, 0.5, 1.5, self.mi, self.mc)
        assert r.shape == (1000,)

    def test_nonnegative(self):
        r = simulate_aftershock_radius(0, 0.5, 1.5, self.mi, self.mc)
        assert np.all(r >= 0)

    def test_finite(self):
        r = simulate_aftershock_radius(0, 0.5, 1.5, self.mi, self.mc)
        assert np.all(np.isfinite(r))

    def test_consistent_with_place(self):
        """Radius from place function should match radius function statistically."""
        np.random.seed(1)
        r_direct = simulate_aftershock_radius(0, 0.5, 1.5, self.mi, self.mc)
        np.random.seed(1)
        x, y = simulate_aftershock_place(0, 0.5, 1.5, self.mi, self.mc)
        r_from_place = np.sqrt(x**2 + y**2)
        assert_allclose(np.median(r_direct), np.median(r_from_place), rtol=0.1)

    def test_larger_magnitude_larger_radius(self):
        mi_small = np.full(2000, 3.0)
        mi_large = np.full(2000, 6.0)
        r_small = simulate_aftershock_radius(1, 0.5, 1.5, mi_small, self.mc)
        r_large = simulate_aftershock_radius(1, 0.5, 1.5, mi_large, self.mc)
        assert np.median(r_large) > np.median(r_small)


# ── inverse_upper_gamma_ext ───────────────────────────────────────────────────

class TestInverseUpperGammaExt:

    def test_positive_a_roundtrip(self):
        """Γ(a, inverse_Γ(a, y)) should recover y for a > 0."""
        a = 0.5
        x = np.array([0.1, 0.5, 1.0, 2.0])
        y = upper_gamma_ext(a, x)
        x_recovered = inverse_upper_gamma_ext(a, y)
        assert_allclose(x_recovered, x, rtol=1e-5)

    def test_negative_a_roundtrip(self):
        """Roundtrip for a < 0 (the numerically inverted branch)."""
        a = -0.5
        x = np.array([0.5, 1.0, 2.0])
        y = upper_gamma_ext(a, x)
        x_recovered = inverse_upper_gamma_ext(a, y)
        assert_allclose(x_recovered, x, rtol=1e-4)

    def test_output_finite_positive_a(self):
        a = 1.0
        y = np.array([0.1, 0.3, 0.5])
        x = inverse_upper_gamma_ext(a, y)
        assert np.all(np.isfinite(x))

    def test_output_positive(self):
        a = 0.5
        y = upper_gamma_ext(a, np.array([0.1, 1.0, 2.0]))
        x = inverse_upper_gamma_ext(a, y)
        assert np.all(x > 0)