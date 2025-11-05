import numpy as np
from scipy.stats import rankdata, norm

class SpearmanRobustPCA:
    """
    Spearman-based Robust PCA

    Key idea:
      - Replace each feature with its ranks (Spearman), optionally map ranks to
        normal scores (Gaussian copula via Blom transform).
      - Compute a correlation matrix from those transformed features.
      - Do eigen-decomposition of that correlation matrix (i.e., PCA on correlation).
      - Project samples by applying the *same* rank/gaussianize transform and then
        using the loadings.

    Parameters
    ----------
    n_components : int or None
        Number of components to keep. If None, keep all.
    gaussianize : bool, default=False
        If True, apply a rank -> normal score transform (Blom) before PCA.
        This often improves linear structure while staying robust to outliers.
    center : {'mean', 'median', None}, default='mean'
        Centering used for per-feature transformed data before computing correlation
        and scores. For Spearman correlation, 'mean' is typical; 'median' is extra-robust.
        If None, no centering (not recommended).
    scale : {'std', 'mad', None}, default='std'
        Scaling used for features before computing correlation and scores.
        'std' matches classical correlation; 'mad' is robust. If None, no scaling.
        (Even with 'std', outliers are heavily down-weighted because we’re using ranks.)
    tie_method : {'average','min','max','dense','ordinal'}, default='average'
        How to handle ties in rank computation. See scipy.stats.rankdata.
    handle_missing : {'drop','impute','pairwise'}, default='impute'
        Missing-data strategy:
          - 'drop': drop any row with NaN.
          - 'impute': impute each feature with its median (computed on non-NaN values) before ranking.
          - 'pairwise': compute pairwise correlations ignoring NaNs per pair (then project to SPD).
    whiten : bool, default=False
        If True, scale components to unit variance in the projected space (like sklearn PCA(whiten=True)).
    random_state : int or None
        Unused directly; kept for API compatibility.
    copy : bool, default=True
        If False, may overwrite input array X (not guaranteed).

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes (loadings) in feature space.
    explained_variance_ : ndarray of shape (n_components,)
        Eigenvalues (variance explained by each component).
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Proportion of variance explained by each component.
    mean_ : ndarray of shape (n_features,)
        Mean (or median) of transformed features used for centering.
    scale_ : ndarray of shape (n_features,)
        Scale (std or MAD) of transformed features used for standardization.
    correlation_ : ndarray of shape (n_features, n_features)
        The Spearman-based correlation matrix used for PCA.
    """

    def __init__(self,
                 n_components=None,
                 gaussianize=False,
                 center='mean',
                 scale='std',
                 tie_method='average',
                 handle_missing='impute',
                 whiten=False,
                 random_state=None,
                 copy=True):
        self.n_components = n_components
        self.gaussianize = gaussianize
        self.center = center
        self.scale = scale
        self.tie_method = tie_method
        self.handle_missing = handle_missing
        self.whiten = whiten
        self.random_state = random_state
        self.copy = copy

    # ---------- utilities ----------
    @staticmethod
    def _mad(x, axis=0, eps=1e-12):
        med = np.nanmedian(x, axis=axis, keepdims=True)
        mad = np.nanmedian(np.abs(x - med), axis=axis, keepdims=True)
        # 1.4826 makes MAD consistent with std under Normal
        return (mad * 1.4826).squeeze(axis=axis) + eps

    @staticmethod
    def _ensure_2d(X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
        return X

    def _rank_transform(self, X):
        """
        Rank transform each column (feature) with chosen tie method.
        Returns fractional ranks in (0,1] if gaussianize=False (scaled by n),
        or normal scores if gaussianize=True via Blom: p = (r - 3/8) / (n + 1/4); z = Phi^{-1}(p).
        """
        X = self._ensure_2d(X)
        n, p = X.shape
        Z = np.empty_like(X, dtype=float)

        for j in range(p):
            xj = X[:, j]
            # ranks that ignore NaNs by ranking only non-NaNs
            mask = ~np.isnan(xj)
            if not np.any(mask):
                # all NaN: fill with NaN
                Z[:, j] = np.nan
                continue
            r = np.full(n, np.nan, dtype=float)
            r[mask] = rankdata(xj[mask], method=self.tie_method)
            if self.gaussianize:
                # Blom transform to normal scores
                m = np.sum(mask)
                # guard for tiny m
                pvals = (r[mask] - 3/8.0) / (m + 0.25)
                pvals = np.clip(pvals, 1e-12, 1 - 1e-12)
                r[mask] = norm.ppf(pvals)
            else:
                # fractional ranks in (0,1]
                m = np.sum(mask)
                r[mask] = r[mask] / m
            Z[:, j] = r
        return Z

    def _center_scale(self, Z, for_scores=False):
        """
        Center and scale features in Z based on self.center / self.scale.
        If for_scores=True, use stored mean_/scale_ learned at fit time.
        """
        if for_scores:
            mean = self.mean_
            scale = self.scale_
        else:
            # compute centering
            if self.center is None:
                mean = np.zeros(Z.shape[1])
            elif self.center == 'mean':
                mean = np.nanmean(Z, axis=0)
            elif self.center == 'median':
                mean = np.nanmedian(Z, axis=0)
            else:
                raise ValueError("center must be None, 'mean', or 'median'")

            # compute scaling
            if self.scale is None:
                scale = np.ones(Z.shape[1])
            elif self.scale == 'std':
                scale = np.nanstd(Z, axis=0, ddof=1)
                scale = np.where(scale <= 1e-12, 1.0, scale)
            elif self.scale == 'mad':
                scale = self._mad(Z, axis=0)
                scale = np.where(scale <= 1e-12, 1.0, scale)
            else:
                raise ValueError("scale must be None, 'std', or 'mad'")

            # store if fitting
            self.mean_ = mean
            self.scale_ = scale

        Zc = (Z - mean) / scale
        return Zc

    @staticmethod
    def _nearest_spd(A, eps=1e-8):
        """Project a symmetric matrix A to the nearest Symmetric Positive Definite."""
        # Symmetrize
        A = (A + A.T) / 2.0
        # Eigen clip
        vals, vecs = np.linalg.eigh(A)
        vals_clipped = np.maximum(vals, eps)
        return (vecs * vals_clipped) @ vecs.T

    def _pairwise_corr(self, Zc):
        """
        Compute pairwise Pearson correlation with NaN handling per pair.
        Zc should already be centered & scaled (per feature).
        """
        p = Zc.shape[1]
        C = np.eye(p)
        for i in range(p):
            zi = Zc[:, i]
            for j in range(i+1, p):
                zj = Zc[:, j]
                mask = ~np.isnan(zi) & ~np.isnan(zj)
                if np.sum(mask) < 2:
                    rho = 0.0
                else:
                    zi0 = zi[mask]
                    zj0 = zj[mask]
                    # since Zc is already standardized per feature, corr = mean(zi*zj)
                    rho = np.sum(zi0 * zj0) / (len(zi0) - 1)
                    # Bound to [-1,1] numerically
                    rho = float(np.clip(rho, -1.0, 1.0))
                C[i, j] = C[j, i] = rho
        # Project to SPD to avoid small negative eigenvalues from pairwise NaN handling
        return self._nearest_spd(C)

    # ---------- main API ----------
    def fit(self, X, y=None):
        """
        Fit the Spearman-based PCA model.

        Returns
        -------
        self
        """
        X = self._ensure_2d(np.array(X, copy=self.copy))
        n, p = X.shape

        # Handle missing values before ranking
        if self.handle_missing == 'drop':
            mask_rows = ~np.isnan(X).any(axis=1)
            X = X[mask_rows]
            n = X.shape[0]
        elif self.handle_missing == 'impute':
            med = np.nanmedian(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(med, inds[1])
        elif self.handle_missing == 'pairwise':
            # leave NaNs in place; pairwise handled later
            pass
        else:
            raise ValueError("handle_missing must be 'drop', 'impute', or 'pairwise'.")

        # Rank (and optional gaussianize)
        Z = self._rank_transform(X)

        # Center & scale transformed features
        Zc = self._center_scale(Z, for_scores=False)

        # Build correlation matrix
        if self.handle_missing == 'pairwise':
            C = self._pairwise_corr(Zc)
        else:
            # classical correlation from standardized Zc (ignoring NaNs as none expected now)
            # Since Zc is standardized per feature, corr = (Zc^T Zc) / (n-1)
            C = (Zc.T @ Zc) / (Zc.shape[0] - 1)
            # numerical clean-up
            C = np.clip(C, -1.0, 1.0)
            C = (C + C.T) / 2.0

        self.correlation_ = C

        # Eigen-decomposition
        evals, evecs = np.linalg.eigh(C)
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]

        if self.n_components is None:
            k = p
        else:
            k = int(self.n_components)
            if k < 1 or k > p:
                raise ValueError("n_components must be in [1, n_features].")

        self.explained_variance_ = evals[:k]
        self.explained_variance_ratio_ = evals[:k] / np.sum(evals)
        # components_ shape (k, p) to match sklearn (rows are components)
        self.components_ = evecs[:, :k].T

        if self.whiten:
            # whiten so projected components have unit variance
            self.components_ = self.components_ / np.sqrt(self.explained_variance_)[:, None]

        # Keep info to re-apply the feature transform at transform() time
        self._fitted_n_features_ = p
        self._fitted_handle_missing_ = self.handle_missing
        self._fitted_gaussianize_ = self.gaussianize
        self._fitted_tie_method_ = self.tie_method

        # Store empirical CDF artifacts for inverse_transform
        self._empirical_sorted_ = np.array([np.sort(X[:, j]) for j in range(p)], dtype=object)
        return self

    def transform(self, X):
        """
        Project X into principal component space (scores).

        Returns
        -------
        T : ndarray of shape (n_samples, n_components)
        """
        if not hasattr(self, "components_"):
            raise RuntimeError("Call fit before transform.")

        X = self._ensure_2d(np.array(X, copy=self.copy))

        # Missing handling during transform:
        if self._fitted_handle_missing_ == 'drop':
            # For unseen data with NaN, we cannot drop rows inside transform cleanly.
            # We impute with training medians as a practical default.
            med = np.array([np.median(s) if len(s) > 0 else np.nan for s in self._empirical_sorted_], dtype=float)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(med, inds[1])
        elif self._fitted_handle_missing_ == 'impute':
            med = np.array([np.median(s) if len(s) > 0 else np.nan for s in self._empirical_sorted_], dtype=float)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(med, inds[1])
        elif self._fitted_handle_missing_ == 'pairwise':
            # For scores, we still need values. Impute with median.
            med = np.array([np.median(s) if len(s) > 0 else np.nan for s in self._empirical_sorted_], dtype=float)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(med, inds[1])

        # Apply the same rank/gaussianize transform
        Z = self._rank_transform(X)
        # Standardize with training centering/scaling
        Zc = self._center_scale(Z, for_scores=True)  # uses mean_/scale_
        # Project
        T = Zc @ self.components_.T
        return T

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, T):
        """
        Approximate inverse transform:
          - Map scores back to Zc via loadings.
          - Undo standardization.
          - If gaussianized: obtain per-feature quantiles via Phi, then
            map quantiles to empirical quantiles of original data.
          - If only ranks used: map fractional ranks back to empirical quantiles.
        Returns
        -------
        X_hat : ndarray of shape (n_samples, n_features)
        """
        if not hasattr(self, "components_"):
            raise RuntimeError("Call fit before inverse_transform.")

        T = np.asarray(T)
        # Reconstruct in standardized-transformed feature space
        if self.whiten:
            # components_ are whitened; reverse whitening to get Zc approx
            # Zc ≈ T @ (components_ * sqrt(evals))
            Zc = T @ (self.components_ * np.sqrt(self.explained_variance_)[:, None])
        else:
            Zc = T @ self.components_

        # Undo standardization
        Z = Zc * self.scale_ + self.mean_

        # Map back to X via inverse of the rank/gaussianize transforms using empirical CDFs
        n, p = Z.shape
        X_hat = np.empty_like(Z)

        for j in range(p):
            sorted_x = self._empirical_sorted_[j]
            m = len(sorted_x)
            if m == 0:
                X_hat[:, j] = np.nan
                continue

            if self.gaussianize:
                # Z is normal scores ~ N(0,1) approx; map to CDF then to empirical quantile
                u = norm.cdf(Z[:, j])
                # clip to (0,1)
                u = np.clip(u, 1e-12, 1 - 1e-12)
            else:
                # Z are fractional ranks in (0,1] (approximately, after undoing center/scale)
                # Coerce to [1/m, 1] to avoid 0
                u = np.clip(Z[:, j], 1.0/m, 1.0)

            # Map u to empirical quantiles by linear interpolation
            # Empirical quantile positions: q_k = (k - 0.5)/m for k=1..m
            qpos = (np.arange(1, m+1) - 0.5) / m
            X_hat[:, j] = np.interp(u, qpos, sorted_x)

        return X_hat
