"""QA independent validation suite for the typhoon CAT model.

Run from anywhere:

    /path/to/python tests/test_validation.py

Only uses numpy / pandas / scipy / matplotlib + stdlib.
All checks are re-derived independently from raw arrays; the model's own
printed self-checks are NOT trusted.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg_mod
import exposure as expo_mod
import financial as fin
import hazard as hz
import vulnerability as vul

RHO = cfg_mod.AIR_DENSITY
HAZ = cfg_mod.HAZARD
STO = cfg_mod.STOCHASTIC
FIN = cfg_mod.FINANCIAL
VUL = cfg_mod.VULNERABILITY


# --------------------------------------------------------------------------- #
# Shared pipeline fixture (built once, reused by all financial tests)
# --------------------------------------------------------------------------- #
class _Pipeline:
    """Recompute the full model chain once, independently of main.py."""

    _cache = None

    @classmethod
    def get(cls):
        if cls._cache is not None:
            return cls._cache
        track = hz.load_lekima_track()
        exposure = expo_mod.load_exposure()
        gust = hz.max_wind_field_over_track(track, exposure.lon, exposure.lat,
                                            HAZ, return_gust=True)
        dist, dp = hz.min_distance_to_track(track, exposure.lon, exposure.lat)
        calib = vul.calibrate_vulnerability(gust, dist, dp, exposure, VUL,
                                            curve="emanuel")
        events = hz.generate_event_set(STO, HAZ)
        g_ev, d_ev, dp_ev = hz.event_set_max_gust(
            events, exposure.lon, exposure.lat, HAZ, STO.batch_size)
        econ_c, ins_c = vul.city_losses(g_ev, d_ev, dp_ev, exposure,
                                        calib.param_after, VUL)
        econ_ev = econ_c.sum(axis=1)
        ins_ev = ins_c.sum(axis=1)
        ylt_econ = fin.build_year_loss_table(econ_ev, events.freq_lambda,
                                             STO.n_simulation_years,
                                             STO.random_seed)
        ylt_ins = fin.build_year_loss_table(ins_ev, events.freq_lambda,
                                            STO.n_simulation_years,
                                            STO.random_seed)
        cls._cache = dict(track=track, exposure=exposure, gust=gust,
                          dist=dist, dp=dp, calib=calib, events=events,
                          econ_ev=econ_ev, ins_ev=ins_ev,
                          ylt_econ=ylt_econ, ylt_ins=ylt_ins)
        return cls._cache


# --------------------------------------------------------------------------- #
# 1. Holland wind field
# --------------------------------------------------------------------------- #
class TestHollandWind(unittest.TestCase):

    def test_profile_peak_at_rmw_and_decay(self):
        """V(r=Rmw) must be the (near-)max; V -> 0 far away; V >= 0; unimodal."""
        pc, pn = 930.0, 1010.0
        dp = pn - pc                      # 80 hPa
        rmw, b, lat = 30.0, 1.6, 28.4
        r = np.linspace(1.0, 1500.0, 3000)
        v = hz.holland_gradient_wind(r, rmw, dp, b, lat)

        self.assertTrue(np.all(v >= 0.0), "negative wind speed found")
        i_peak = int(np.argmax(v))
        self.assertAlmostEqual(r[i_peak], rmw, delta=5.0,
                               msg=f"peak at {r[i_peak]:.1f} km, expected ~{rmw}")
        # analytic cyclostrophic max sqrt(B*dp/(rho*e)); Coriolis makes it lower
        v_theory = np.sqrt(b * dp * 100.0 / (RHO * np.e))
        self.assertLess(abs(v[i_peak] - v_theory) / v_theory, 0.05,
                        f"peak {v[i_peak]:.2f} vs theory {v_theory:.2f}")
        self.assertLess(v[-1], 0.15 * v[i_peak], "no far-field decay")
        # unimodal: strictly non-increasing after the peak (tolerance for float)
        dv = np.diff(v[i_peak:])
        self.assertTrue(np.all(dv <= 1e-9), "profile not unimodal after peak")
        # increasing before peak
        self.assertTrue(np.all(np.diff(v[: i_peak + 1]) >= -1e-9),
                        "profile not increasing before peak")

    def test_b_parameter_clipped(self):
        b = hz.holland_b_physical(np.array([10.0, 200.0]),
                                  np.array([100.0, 5.0]), HAZ)
        self.assertTrue(np.all(b >= HAZ.holland_b_min - 1e-12))
        self.assertTrue(np.all(b <= HAZ.holland_b_max + 1e-12))
        # physically consistent B reproduces vmax at Rmw (when unclipped)
        vmax_g = 60.0
        dp = 80.0
        b_mid = hz.holland_b_physical(np.array([vmax_g]), np.array([dp]), HAZ)[0]
        b_manual = RHO * np.e * vmax_g ** 2 / (dp * 100.0)
        self.assertAlmostEqual(b_mid, np.clip(b_manual, HAZ.holland_b_min,
                                              HAZ.holland_b_max), places=10)


# --------------------------------------------------------------------------- #
# 2 & 3 & 4. EP curve / AAL / VaR-TVaR
# --------------------------------------------------------------------------- #
class TestEPandRiskMetrics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p = _Pipeline.get()
        cls.m_econ = fin.compute_risk_metrics(cls.p["ylt_econ"], FIN)
        cls.m_ins = fin.compute_risk_metrics(cls.p["ylt_ins"], FIN)

    def test_oep_monotone_and_aep_ge_oep(self):
        rps = list(FIN.pml_return_periods)
        oep = [self.m_econ.oep_pml[r] for r in rps]
        aep = [self.m_econ.aep_pml[r] for r in rps]
        self.assertTrue(all(a <= b + 1e-9 for a, b in zip(oep, oep[1:])),
                        f"OEP not monotone: {oep}")
        for r, o, a in zip(rps, oep, aep):
            self.assertGreaterEqual(a + 1e-9, o, f"AEP < OEP at RP={r}")

    def test_return_period_definition_self_consistent(self):
        """PML(T) uses quantile 1-1/T; verify empirically P(L >= PML(T)) ~ 1/T."""
        occ = self.p["ylt_econ"].occurrence
        for t in (10.0, 100.0, 250.0):
            pml = fin.pml_at_return_periods(occ, [t])[0]
            p_exc = np.mean(occ > pml)
            self.assertLess(abs(p_exc - 1.0 / t) * t, 0.15,
                            f"RP={t}: empirical exc prob {p_exc:.5f} vs {1/t:.5f}")
        # ep_curve internal consistency: rp = 1/prob exactly
        x, prob, rp = fin.ep_curve(occ)
        np.testing.assert_allclose(rp * prob, 1.0, rtol=1e-12)
        self.assertTrue(np.all(np.diff(x) <= 1e-9), "ep_curve losses not desc")

    def test_aal_two_independent_paths(self):
        """AAL == mean(aggregate) and ~ lambda * mean(event loss) (<5%)."""
        ylt = self.p["ylt_econ"]
        aal_reported = ylt.aal
        aal_path1 = float(np.mean(ylt.aggregate))
        aal_path2 = float(ylt.freq_lambda * np.mean(self.p["econ_ev"]))
        self.assertAlmostEqual(aal_reported, aal_path1, places=9)
        rel = abs(aal_path1 - aal_path2) / aal_path2
        self.assertLess(rel, 0.05,
                        f"AAL mismatch: YLT {aal_path1:.2f} vs "
                        f"lambda*E[L] {aal_path2:.2f} ({rel:.2%})")
        # aggregate must equal sum of event losses per year
        recon = np.bincount(ylt.event_year, weights=ylt.event_loss,
                            minlength=ylt.n_years)
        np.testing.assert_allclose(recon, ylt.aggregate, rtol=1e-12)

    def test_var_tvar_recomputed(self):
        agg = self.p["ylt_ins"].aggregate
        v99, t99 = fin.var_tvar(agg, 0.99)
        v995, _ = fin.var_tvar(agg, 0.995)
        # own recomputation with plain numpy
        my_v99 = float(np.quantile(agg, 0.99))
        my_t99 = float(agg[agg > my_v99].mean())
        self.assertAlmostEqual(v99, my_v99, places=9)
        self.assertAlmostEqual(t99, my_t99, places=9)
        self.assertGreaterEqual(t99, v99, "TVaR99 < VaR99")
        self.assertGreaterEqual(v995, v99, "VaR99.5 < VaR99")
        # C-ROSS capital = VaR99.5 - AAL
        m = fin.compute_risk_metrics(self.p["ylt_ins"], FIN)
        self.assertAlmostEqual(m.c_ross_capital,
                               v995 - float(np.mean(agg)), places=9)


# --------------------------------------------------------------------------- #
# 5. Reinsurance layers
# --------------------------------------------------------------------------- #
class TestReinsurance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p = _Pipeline.get()
        cls.layers = fin.build_reinsurance_program(cls.p["ylt_ins"], FIN)

    def test_layer_el_recomputed_from_samples(self):
        ylt = self.p["ylt_ins"]
        for ly in self.layers:
            rec = np.minimum(np.maximum(ylt.event_loss - ly.attachment, 0.0),
                             ly.limit)
            annual = np.bincount(ylt.event_year, weights=rec,
                                 minlength=ylt.n_years)
            self.assertAlmostEqual(ly.expected_loss, float(annual.mean()),
                                   places=9, msg=ly.name)
            self.assertAlmostEqual(ly.std_recovery,
                                   float(annual.std(ddof=1)), places=9)
            # premium formula
            loaded = (annual.mean() + FIN.layer_sd_load * annual.std(ddof=1)) \
                * (1.0 + FIN.layer_expense_ratio)
            self.assertAlmostEqual(ly.loaded_premium, float(loaded), places=9)
            self.assertAlmostEqual(ly.rate_on_line,
                                   ly.loaded_premium / ly.limit, places=12)

    def test_market_regularities(self):
        els = [ly.el_rate for ly in self.layers]
        mults = [ly.multiple for ly in self.layers]
        atts = [ly.attachment for ly in self.layers]
        self.assertTrue(all(a < b for a, b in zip(atts, atts[1:])),
                        "attachments not increasing")
        self.assertTrue(all(a > b for a, b in zip(els, els[1:])),
                        f"EL rate not decreasing with attachment: {els}")
        self.assertTrue(all(a < b for a, b in zip(mults, mults[1:])),
                        f"multiple not increasing with attachment: {mults}")
        for ly in self.layers:
            self.assertGreater(ly.rate_on_line, ly.el_rate,
                               "ROL <= EL rate (no risk load?)")
            self.assertLess(ly.rate_on_line, 1.0, "ROL >= 100%: unit error")


# --------------------------------------------------------------------------- #
# 6. CAT bond pricing
# --------------------------------------------------------------------------- #
class TestCatBond(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p = _Pipeline.get()
        ylt = cls.p["ylt_ins"]
        cls.a = fin._nice_round(float(fin.pml_at_return_periods(
            ylt.occurrence, [FIN.catbond_attach_rp])[0]))
        cls.e = fin._nice_round(float(fin.pml_at_return_periods(
            ylt.occurrence, [FIN.catbond_exhaust_rp])[0]))
        cls.bond = fin.price_cat_bond(ylt.occurrence, cls.a, cls.e,
                                      "industry index", FIN)

    def test_spread_above_el_and_in_market_range(self):
        b = self.bond
        self.assertGreater(b.spread_lane, b.expected_loss)
        self.assertGreater(b.spread_wang, b.expected_loss)
        for s in (b.spread_lane, b.spread_wang):
            self.assertGreaterEqual(s * 1e4, 200.0, f"{s*1e4:.0f}bp < 200bp")
            self.assertLessEqual(s * 1e4, 1500.0, f"{s*1e4:.0f}bp > 1500bp")

    def test_el_pfl_cel_recomputed(self):
        ylt = self.p["ylt_ins"]
        lr = np.clip((ylt.occurrence - self.a) / (self.e - self.a), 0.0, 1.0)
        self.assertAlmostEqual(self.bond.expected_loss, float(lr.mean()),
                               places=12)
        self.assertAlmostEqual(self.bond.prob_first_loss,
                               float(np.mean(lr > 0)), places=12)
        cel = lr.mean() / np.mean(lr > 0)
        self.assertAlmostEqual(self.bond.cond_expected_loss, float(cel),
                               places=12)
        # Lane formula by hand
        eer = FIN.lane_gamma * self.bond.prob_first_loss ** FIN.lane_alpha \
            * cel ** FIN.lane_beta
        self.assertAlmostEqual(self.bond.spread_lane,
                               float(lr.mean() + eer), places=12)

    def test_wang_transform_direction_and_value(self):
        """lambda > 0 must be MORE conservative (spread > EL), increasing in lambda."""
        lr = self.bond.loss_ratios
        w0 = fin.wang_transform_price(lr, 0.0)
        w_mkt = fin.wang_transform_price(lr, FIN.wang_lambda_market)
        w_hi = fin.wang_transform_price(lr, 1.0)
        el = float(lr.mean())
        self.assertLess(abs(w0 - el) / max(el, 1e-9), 0.05,
                        f"lambda=0 should recover EL: {w0:.6f} vs {el:.6f}")
        self.assertGreater(w_mkt, w0, "lambda>0 did not raise the price")
        self.assertGreater(w_hi, w_mkt, "price not increasing in lambda")
        # independent re-implementation via survival distortion
        # S*(x) = Phi(Phi^-1(S(x)) + lam) integrated over [0, 1]
        lam = FIN.wang_lambda_market
        grid = np.linspace(0.0, 1.0, 1001)
        surv = 1.0 - np.searchsorted(np.sort(lr), grid, side="right") / lr.size
        surv = np.clip(surv, 1e-12, 1 - 1e-12)
        surv_star = norm.cdf(norm.ppf(surv) + lam)
        mine = float(np.trapezoid(surv_star, grid))
        self.assertLess(abs(mine - w_mkt) / max(mine, 1e-9), 0.02,
                        f"Wang spread mismatch: mine {mine:.6f} vs {w_mkt:.6f}")

    def test_coupon_equals_rf_plus_spread(self):
        self.assertAlmostEqual(self.bond.coupon_lane,
                               FIN.risk_free_rate + self.bond.spread_lane,
                               places=12)
        self.assertAlmostEqual(self.bond.coupon_wang,
                               FIN.risk_free_rate + self.bond.spread_wang,
                               places=12)


# --------------------------------------------------------------------------- #
# 7. Basis risk
# --------------------------------------------------------------------------- #
class TestBasisRisk(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p = _Pipeline.get()
        ev = cls.p["events"]
        box = fin.design_location_box(
            ev.landfall_lat, cls.p["ins_ev"], n_bins=10,
            lat_range=(STO.landfall_lat_min, STO.landfall_lat_max))
        cls.payout = fin.parametric_payout_ratio(
            ev.landfall_pc, FIN, ev.landfall_lat, box)
        cls.res = fin.analyse_basis_risk(cls.p["ins_ev"], cls.payout)

    def test_hedge_effectiveness_recomputed_and_bounded(self):
        l = self.p["ins_ev"]
        pr = self.payout
        q = np.cov(l, pr, ddof=1)[0, 1] / np.var(pr)
        he = 1.0 - np.var(l - q * pr) / np.var(l)
        self.assertAlmostEqual(self.res.hedge_effectiveness, float(he),
                               places=6)
        self.assertGreaterEqual(he, 0.0)
        self.assertLessEqual(he, 1.0)
        # at optimal q, HE should equal corr^2
        self.assertLess(abs(he - self.res.correlation ** 2), 1e-3)

    def test_pearson_recomputed(self):
        corr = float(np.corrcoef(self.p["ins_ev"], self.payout)[0, 1])
        self.assertAlmostEqual(self.res.correlation, corr, places=10)
        self.assertGreater(corr, 0.0)


# --------------------------------------------------------------------------- #
# 8. Calibration is real (anti-cheating test) - HIGHEST PRIORITY
# --------------------------------------------------------------------------- #
class TestCalibrationNotFaked(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p = _Pipeline.get()

    def test_loss_recomputed_outside_calibrator(self):
        """Plug the calibrated V_half back into city_losses ourselves."""
        p = self.p
        econ, _ = vul.city_losses(p["gust"], p["dist"], p["dp"],
                                  p["exposure"], p["calib"].param_after, VUL)
        total = float(np.sum(econ))
        self.assertLess(abs(total - VUL.lekima_actual_loss) / VUL.lekima_actual_loss,
                        1e-4,
                        f"independent recomputation {total:.2f} != 537.2")
        # and the reported loss_after must equal this independent value
        self.assertAlmostEqual(total, p["calib"].loss_after, places=6)

    def test_calibration_follows_a_different_target(self):
        """If we change the target, the calibrated parameter MUST move and
        the loss MUST hit the new target. Hard-coded returns would fail."""
        p = self.p
        for target in (400.0, 700.0):
            c = vul.calibrate_vulnerability(p["gust"], p["dist"], p["dp"],
                                            p["exposure"], VUL,
                                            curve="emanuel",
                                            target_loss=target)
            econ, _ = vul.city_losses(p["gust"], p["dist"], p["dp"],
                                      p["exposure"], c.param_after, VUL)
            got = float(np.sum(econ))
            self.assertLess(abs(got - target) / target, 1e-3,
                            f"target {target}: got {got:.2f}")
            self.assertNotAlmostEqual(c.param_after,
                                      p["calib"].param_after, places=2,
                                      msg="parameter did not move with target")
        # monotonicity: higher target -> lower V_half (more vulnerable)
        c_lo = vul.calibrate_vulnerability(p["gust"], p["dist"], p["dp"],
                                           p["exposure"], VUL,
                                           curve="emanuel", target_loss=400.0)
        c_hi = vul.calibrate_vulnerability(p["gust"], p["dist"], p["dp"],
                                           p["exposure"], VUL,
                                           curve="emanuel", target_loss=700.0)
        self.assertGreater(c_lo.param_after, c_hi.param_after,
                           "V_half not monotone decreasing in target loss")

    def test_loss_monotone_in_v_half(self):
        """Loss must strictly decrease as V_half rises (brentq precondition)."""
        p = self.p
        vals = []
        for vh in (100.0, 150.0, 200.0, 300.0):
            econ, _ = vul.city_losses(p["gust"], p["dist"], p["dp"],
                                      p["exposure"], vh, VUL)
            vals.append(float(np.sum(econ)))
        self.assertTrue(all(a > b for a, b in zip(vals, vals[1:])),
                        f"loss not monotone in V_half: {vals}")


# --------------------------------------------------------------------------- #
# Reproducibility of the event set / YLT under fixed seed
# --------------------------------------------------------------------------- #
class TestSeedStability(unittest.TestCase):

    def test_event_set_deterministic(self):
        e1 = hz.generate_event_set(STO, HAZ)
        e2 = hz.generate_event_set(STO, HAZ)
        np.testing.assert_array_equal(e1.landfall_dp, e2.landfall_dp)
        np.testing.assert_array_equal(e1.lon, e2.lon)

    def test_ylt_deterministic(self):
        losses = np.abs(np.random.default_rng(1).normal(10, 5, 500))
        y1 = fin.build_year_loss_table(losses, 3.2, 20000, 42)
        y2 = fin.build_year_loss_table(losses, 3.2, 20000, 42)
        np.testing.assert_array_equal(y1.aggregate, y2.aggregate)


if __name__ == "__main__":
    unittest.main(verbosity=2)
