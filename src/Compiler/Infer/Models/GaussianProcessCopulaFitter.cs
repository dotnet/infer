// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Distributions.Copulas;
using Microsoft.ML.Probabilistic.Distributions.Copulas.Vine;
using Microsoft.ML.Probabilistic.Distributions.Kernels;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Models
{
    /// <summary>
    /// Fits a single conditional bivariate copula <c>c(u, v | z)</c> for a vine edge with a
    /// sparse GP latent function and Expectation Propagation (the GPVINE model of Lopez-Paz et
    /// al., 2013, Sec. 3). This is the modelling-layer implementation of the Runtime
    /// <see cref="IConditionalCopulaFitter"/> interface that the vine recursion injects.
    /// </summary>
    /// <remarks>
    /// The model mirrors the Gaussian-process classifier tutorial: a <see cref="SparseGP"/> prior
    /// over <c>f</c>, <c>score = f(z)</c> via <see cref="Variable.FunctionEvaluate"/>, and the
    /// observed copula pair as the likelihood through <see cref="Variable.BivariateCopula"/>
    /// (which folds in the link <c>tau = 2*Phi(score) - 1</c>). The GP mean is initialised to
    /// <c>Phi^{-1}((tau_MLE + 1)/2)</c> (Sec. 4) and the inducing inputs to a subset of the
    /// conditioning data (Sec. 3.1).
    /// </remarks>
    public class GaussianProcessCopulaFitter : IConditionalCopulaFitter
    {
        private readonly InferenceEngine engine;

        /// <summary>Number of inducing inputs (pseudo-inputs) for the FITC sparse GP (paper: 20).</summary>
        public int NumInducing { get; set; } = 20;

        /// <summary>
        /// Log length-scale used to initialise every ARD kernel dimension. The default suits
        /// conditioning inputs that are copula pseudo-observations in (0, 1).
        /// </summary>
        public double LogLengthScale { get; set; } = -1.5;

        /// <summary>Log signal standard deviation of the ARD kernel.</summary>
        public double LogSignalSd { get; set; } = 0.0;

        /// <summary>
        /// Log standard deviation of an added white-noise kernel component. This regularises the
        /// inducing-point covariance (keeping it positive definite) and acts as the GP nugget.
        /// </summary>
        public double LogNoiseSd { get; set; } = System.Math.Log(0.2);

        /// <summary>Number of EP iterations per edge fit.</summary>
        public int NumberOfIterations { get; set; } = 15;

        /// <summary>The log model evidence of the most recent <see cref="Fit"/> (NaN if not computed).</summary>
        public double LastLogEvidence { get; private set; } = double.NaN;

        /// <summary>
        /// Creates a fitter. If no engine is supplied, a non-verbose Expectation Propagation
        /// engine is used.
        /// </summary>
        public GaussianProcessCopulaFitter(InferenceEngine engine = null)
        {
            this.engine = engine ?? new InferenceEngine(new ExpectationPropagation()) { ShowProgress = false };
        }

        /// <inheritdoc/>
        public IConditionalCopulaPosterior Fit(double[] u, double[] v, double[][] z, CopulaFamily family)
        {
            if (u == null) throw new ArgumentNullException(nameof(u));
            if (v == null) throw new ArgumentNullException(nameof(v));
            if (z == null) throw new ArgumentNullException(nameof(z));
            int n = u.Length;
            if (n == 0) throw new ArgumentException("No observations.", nameof(u));
            int dim = z[0].Length;
            if (dim == 0) throw new ArgumentException("Conditioning set is empty; fit unconditionally instead.", nameof(z));

            Vector[] inputs = z.Select(Vector.FromArray).ToArray();
            Vector[] pairs = new Vector[n];
            for (int i = 0; i < n; i++)
                pairs[i] = Vector.FromArray(u[i], v[i]);

            // GP mean initialised from the unconditional Kendall's-tau MLE (Sec. 4).
            double tauMle = KendallTau.Compute(u, v);
            double meanInit = MMath.NormalCdfInv(0.5 * (Clamp(tauMle) + 1.0));

            // Model: SparseGP prior -> score = f(z) -> observed copula pair likelihood.
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);

            Variable<SparseGP> prior = Variable.New<SparseGP>().Named("copulaPrior");
            Variable<IFunction> f = Variable<IFunction>.Random(prior).Named("f");

            VariableArray<Vector> zVar = Variable.Observed(inputs).Named("z");
            Range j = zVar.Range.Named("j");
            Variable<double> score = Variable.FunctionEvaluate(f, zVar[j]).Named("score");

            VariableArray<Vector> uv = Variable.Observed(pairs, j).Named("uv");
            uv[j] = Variable.BivariateCopula(score, (int)family);

            block.CloseBlock();

            var ard = new ARD(Enumerable.Repeat(LogLengthScale, dim).ToArray(), LogSignalSd);
            var kernel = new SummationKernel(ard) + new WhiteNoise(LogNoiseSd);
            var gp = new GaussianProcess(new ConstantFunction(meanInit), kernel);
            prior.ObservedValue = new SparseGP(new SparseGPFixed(gp, SelectInducing(inputs)));

            engine.NumberOfIterations = NumberOfIterations;
            LastLogEvidence = engine.Infer<Bernoulli>(evidence).LogOdds;
            SparseGP posterior = engine.Infer<SparseGP>(f);
            return new SparseGPCopulaPosterior(posterior);
        }

        /// <summary>Selects up to <see cref="NumInducing"/> inducing inputs as an evenly spaced subset.</summary>
        private Vector[] SelectInducing(Vector[] inputs)
        {
            int n = inputs.Length;
            if (n <= NumInducing)
                return inputs.Select(p => Vector.Copy(p)).ToArray();
            Vector[] basis = new Vector[NumInducing];
            for (int b = 0; b < NumInducing; b++)
            {
                int idx = (int)((long)b * (n - 1) / (NumInducing - 1));
                basis[b] = Vector.Copy(inputs[idx]);
            }
            return basis;
        }

        private static double Clamp(double tau, double eps = 1e-4)
        {
            if (tau < -1.0 + eps) return -1.0 + eps;
            if (tau > 1.0 - eps) return 1.0 - eps;
            return tau;
        }
    }
}
