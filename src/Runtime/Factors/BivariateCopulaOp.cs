// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Distributions.Copulas;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <summary>
    /// Expectation Propagation message operator for the
    /// <see cref="CopulaFactor.BivariateCopula(double, IBivariateCopula)"/> factor, the core
    /// inference component of the GPVINE method (Lopez-Paz, Hernandez-Lobato and Ghahramani,
    /// ICML 2013).
    /// </summary>
    /// <remarks>
    /// <para>
    /// For an observed pseudo-observation pair <c>(u, v)</c> the factor contributes the
    /// copula likelihood <c>c(u, v | tau)</c> with <c>tau = g(score) = 2*Phi(score) - 1</c>
    /// (Sec. 3) as a non-conjugate term on the single Gaussian latent <c>score = f(z)</c>.
    /// Because the link <c>g</c> is folded inside the factor, EP only ever sees a smooth
    /// positive likelihood on one Gaussian variable - structurally identical to the
    /// situation handled by <see cref="ExpOp"/> / <c>LogisticOp</c>.
    /// </para>
    /// <para>
    /// The outgoing message to <c>score</c> is obtained by moment matching: with cavity
    /// <c>N(s; m, v)</c> the tilted distribution is <c>N(s; m, v) c(u, v | g(s))</c>, whose
    /// first two moments are computed by Gauss-Hermite quadrature (in log space, with a
    /// log-sum-exp accumulation for numerical stability). The projected Gaussian divided by
    /// the cavity is the message; the log normalizer of the tilted integral is the evidence
    /// contribution used for hyperparameter tuning and family selection (Sec. 4).
    /// </para>
    /// </remarks>
    [FactorMethod(typeof(CopulaFactor), "BivariateCopula")]
    [Quality(QualityBand.Experimental)]
    public static class BivariateCopulaOp
    {
        /// <summary>
        /// Initial node count for the adaptive Clenshaw-Curtis quadrature used for moment matching.
        /// </summary>
        public static int QuadratureNodeCount = 64;

        /// <summary>Relative tolerance for the adaptive quadrature.</summary>
        public static double QuadratureRelTol = 1e-10;

        /// <summary>Whether to force the outgoing message to be proper (non-negative precision).</summary>
        public static bool ForceProper = true;

        /// <summary>
        /// EP message to the latent <c>score</c> (eq. 8 factor, moment-matched).
        /// </summary>
        /// <param name="pair">Observed pseudo-observation pair (u, v) in (0, 1)^2.</param>
        /// <param name="score">Incoming message from <c>score</c> (the cavity).</param>
        /// <param name="copula">The copula family evaluator.</param>
        /// <param name="result">Previous outgoing message; used only as a quadrature proposal.</param>
        /// <returns>The outgoing message to <c>score</c>.</returns>
        public static Gaussian ScoreAverageConditional(Vector pair, [Proper] Gaussian score, IBivariateCopula copula, Gaussian result)
        {
            // Without a proper, finite-variance cavity there is nothing to moment-match against.
            if (score.IsUniform() || score.IsPointMass)
                return Gaussian.Uniform();

            Gaussian proposal = (result != null && result.IsProper() && !result.IsUniform()) ? score * result : score;
            ComputeMoments(pair, score, proposal, copula, out _, out double mean, out double variance);

            if (variance <= 0.0)
            {
                // Quadrature can collapse the variance when nodes are too far apart; fall back
                // to an uninformative message rather than an improper one.
                return Gaussian.Uniform();
            }

            Gaussian projected = Gaussian.FromMeanAndVariance(mean, variance);
            result = new Gaussian();
            result.SetToRatio(projected, score, ForceProper);
            if (double.IsNaN(result.Precision) || double.IsNaN(result.MeanTimesPrecision))
                return Gaussian.Uniform();
            return result;
        }

        /// <summary>
        /// Initializer for <see cref="ScoreAverageConditional"/>.
        /// </summary>
        [Skip]
        public static Gaussian ScoreAverageConditionalInit()
        {
            return Gaussian.Uniform();
        }

        /// <summary>
        /// Evidence contribution: <c>log int N(s; m, v) c(u, v | g(s)) ds</c>.
        /// </summary>
        /// <param name="pair">Observed pseudo-observation pair (u, v).</param>
        /// <param name="score">Incoming message from <c>score</c> (the cavity).</param>
        /// <param name="copula">The copula family evaluator.</param>
        /// <param name="to_score">Current outgoing message; used only as a quadrature proposal.</param>
        public static double LogAverageFactor(Vector pair, Gaussian score, IBivariateCopula copula, Gaussian to_score)
        {
            if (score.IsPointMass)
                return copula.LogDensity(pair[0], pair[1], 2.0 * MMath.NormalCdf(score.Point) - 1.0);
            if (score.IsUniform())
                return 0.0;
            Gaussian proposal = (to_score != null && to_score.IsProper() && !to_score.IsUniform()) ? score * to_score : score;
            ComputeMoments(pair, score, proposal, copula, out double logZ, out _, out _);
            return logZ;
        }

        /// <summary>
        /// Evidence ratio for EP. The factor output (the pair) is observed, so as with
        /// the deterministic-output overloads of <see cref="ExpOp"/> / <c>LogisticOp</c>
        /// this equals <see cref="LogAverageFactor"/>.
        /// </summary>
        public static double LogEvidenceRatio(Vector pair, Gaussian score, IBivariateCopula copula, [Fresh] Gaussian to_score)
        {
            return LogAverageFactor(pair, score, copula, to_score);
        }

        /// <summary>
        /// Moment-matches the tilted distribution <c>N(s; m, v) c(u, v | g(s))</c> by
        /// Gauss-Hermite quadrature, returning the log normalizer and the first two moments.
        /// </summary>
        /// <param name="pair">Observed pseudo-observation pair (u, v).</param>
        /// <param name="score">The cavity N(s; m, v) - the measure of the integral.</param>
        /// <param name="proposal">Gaussian used to place the quadrature nodes (e.g. score*to_score).</param>
        /// <param name="copula">The copula family evaluator.</param>
        /// <param name="logZ"><c>log int score(s) c(u, v | g(s)) ds</c>.</param>
        /// <param name="mean">Mean of the tilted distribution.</param>
        /// <param name="variance">Variance of the tilted distribution.</param>
        private static void ComputeMoments(
            Vector pair, Gaussian score, Gaussian proposal, IBivariateCopula copula,
            out double logZ, out double mean, out double variance)
        {
            double u = pair[0];
            double v = pair[1];
            proposal.GetMeanAndVariance(out double mProp, out double vProp);
            if (vProp <= 0.0)
                score.GetMeanAndVariance(out mProp, out vProp);
            double sc = Math.Sqrt(vProp);

            // log of the tilted, unnormalized integrand: cavity(s) * c(u, v | g(s)).
            // The copula likelihood can blow up as the latent pushes tau -> +-1 (centered
            // observations), so the integrand is heavy-tailed; adaptive Clenshaw-Curtis
            // with its 1/tan transform resolves this far better than fixed Gauss-Hermite.
            double LogTilted(double s)
            {
                double tau = 2.0 * MMath.NormalCdf(s) - 1.0;
                return score.GetLogProb(s) + copula.LogDensity(u, v, tau);
            }

            // Integrate in standardized coordinates z, with s = sc*z + mProp, so the mass is
            // centered near z = 0 (Clenshaw-Curtis concentrates nodes at the origin). An offset
            // keeps the exponentials in range; it is folded back into logZ analytically.
            double offset = LogTilted(mProp);
            double G(double z) => Math.Exp(LogTilted(sc * z + mProp) - offset);
            double zStd = Quadrature.AdaptiveClenshawCurtis(z => G(z), 1.0, QuadratureNodeCount, QuadratureRelTol);
            double zMean = Quadrature.AdaptiveClenshawCurtis(z => z * G(z), 1.0, QuadratureNodeCount, QuadratureRelTol);
            double zMean2 = Quadrature.AdaptiveClenshawCurtis(z => z * z * G(z), 1.0, QuadratureNodeCount, QuadratureRelTol);

            if (zStd <= 0.0 || double.IsNaN(zStd))
            {
                logZ = double.NegativeInfinity;
                mean = mProp;
                variance = vProp;
                return;
            }

            double meanZ = zMean / zStd;
            double varZ = zMean2 / zStd - meanZ * meanZ;
            mean = mProp + sc * meanZ;
            variance = vProp * varZ;
            // Z = int exp(LogTilted(s)) ds = sc * exp(offset) * zStd.
            logZ = Math.Log(sc) + offset + Math.Log(zStd);
        }
    }
}
