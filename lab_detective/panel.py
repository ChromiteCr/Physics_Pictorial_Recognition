"""特征面板：只用可观测的统计量描述一组线性化数据，所有对比组拿到同一份。

截距 z、斜率比及其 z（理想斜率已知时）、曲率 z、残差散布（χ²/dof 偏大或偏小）、按试次顺序的漂移。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

from .labsim.core import LinearData

NAN = float("nan")


@dataclass
class Panel:
    n: int
    sufficient: bool
    slope: float = NAN
    slope_se: float = NAN
    slope_ratio: float = NAN
    slope_ratio_z: float = NAN
    intercept: float = NAN
    intercept_se: float = NAN
    intercept_z: float = NAN
    curvature_z: float = NAN
    chi2: float = NAN
    dof: int = 0
    chi2_dof: float = NAN
    scatter_p_high: float = NAN
    scatter_p_low: float = NAN
    drift_rho: float = NAN
    drift_p: float = NAN
    signatures: tuple[str, ...] = ("none",)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["signatures"] = list(self.signatures)
        return d


def _wls(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    XtW = X.T * w
    cov = np.linalg.inv(XtW @ X)
    beta = cov @ (XtW @ y)
    resid = y - X @ beta
    return beta, cov, resid


def compute_panel(lin: LinearData, z_crit: float = 2.0, alpha: float = 0.05) -> Panel:
    x, y, sy = np.asarray(lin.x, float), np.asarray(lin.y, float), np.asarray(lin.sy, float)
    n = x.size
    if n < 2 or np.ptp(x) == 0:
        return Panel(n=n, sufficient=False)
    w = 1.0 / sy**2
    X = np.column_stack([x, np.ones(n)])
    beta, cov, resid = _wls(X, y, w)
    a, b = beta
    sa, sb = np.sqrt(np.diag(cov))
    p = Panel(n=n, sufficient=n >= 3)
    p.slope, p.slope_se = float(a), float(sa)
    p.intercept, p.intercept_se = float(b), float(sb)
    p.intercept_z = float((b - lin.ideal_intercept) / sb)
    if lin.ideal_slope is not None and lin.ideal_slope != 0:
        p.slope_ratio = float(a / lin.ideal_slope)
        p.slope_ratio_z = float((a - lin.ideal_slope) / sa)
    sig: list[str] = []
    resid_for_scatter, dof_scatter = resid, n - 2
    if n >= 4:
        xc = x - np.average(x, weights=w)
        Xq = np.column_stack([xc**2, xc, np.ones(n)])
        bq, cq, rq = _wls(Xq, y, w)
        p.curvature_z = float(bq[0] / np.sqrt(cq[0, 0]))
        if abs(p.curvature_z) > z_crit:
            sig.append("curve_up" if p.curvature_z > 0 else "curve_down")
            resid_for_scatter, dof_scatter = rq, n - 3
    chi2 = float(np.sum(resid_for_scatter**2 * w))
    p.chi2, p.dof = chi2, int(dof_scatter)
    if dof_scatter >= 1:
        p.chi2_dof = chi2 / dof_scatter
        p.scatter_p_high = float(stats.chi2.sf(chi2, dof_scatter))
        p.scatter_p_low = float(stats.chi2.cdf(chi2, dof_scatter))
    if n >= 3:
        if abs(p.intercept_z) > z_crit and "curve_up" not in sig and "curve_down" not in sig:
            sig.append("intercept_pos" if p.intercept_z > 0 else "intercept_neg")
        if lin.ideal_slope is not None and abs(p.slope_ratio_z) > z_crit:
            sig.append("slope_high" if p.slope_ratio > 1 else "slope_low")
        if dof_scatter >= 2 and p.scatter_p_high < alpha:
            sig.append("scatter_excess")
        if dof_scatter >= 3 and p.scatter_p_low < alpha:
            sig.append("scatter_deficit")
    if n >= 5:
        rho, pv = stats.spearmanr(np.asarray(lin.order, float), resid_for_scatter)
        p.drift_rho, p.drift_p = float(rho), float(pv)
        if pv < alpha and abs(rho) >= 0.8:
            sig.append("drift")
    p.signatures = tuple(sig) if sig else ("none",)
    return p
