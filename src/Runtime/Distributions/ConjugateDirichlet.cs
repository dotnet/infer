// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Runtime.Serialization;

    using Factors;
    using Factors.Attributes;
    using Math;
    using Microsoft.ML.Probabilistic.Serialization;
    using Utilities;
    
    /// <summary>
    /// Represents the distribution proportion to x^{Shape-1} exp(-Rate*x) / B(x,D)^K
    /// where B(x,D)=Gamma(x)^D/Gamma(D*x)
    /// </summary>
    [Serializable]
    [DataContract]
    [Quality(QualityBand.Experimental)]
    public struct ConjugateDirichlet : IDistribution<double>, //ICursor,
                                       SettableTo<ConjugateDirichlet>, SettableToProduct<ConjugateDirichlet>, Diffable, SettableToUniform,
                                       SettableToRatio<ConjugateDirichlet>, SettableToPower<ConjugateDirichlet>, SettableToWeightedSum<ConjugateDirichlet>,
                                       Sampleable<double>, CanGetMean<double>, CanGetVariance<double>,
                                       CanGetMeanAndVarianceOut<double, double>, CanSetMeanAndVariance<double, double>,
                                       CanGetLogAverageOf<ConjugateDirichlet>, CanGetLogAverageOfPower<ConjugateDirichlet>,
                                       CanGetAverageLog<ConjugateDirichlet>, CanGetLogNormalizer
    {
        /// <summary>
        /// Error message for proposal distribution
        /// </summary>
        private const string errorMessage = "Error calculating Conjugate Dirichlet proposal distribution";

        /// <summary>
        /// Rate parameter for the distribution
        /// </summary>
        [DataMember]
        public double Rate;

        /// <summary>
        /// Shape parameter for the distribution
        /// </summary>
        [DataMember]
        public double Shape;

        /// <summary>
        /// Parameter D for the distribution
        /// </summary>
        [DataMember]
        public double D;

        /// <summary>
        /// Parameter K for the distribution
        /// </summary>
        [DataMember]
        public double K;

        /// <summary>
        /// Gets the expected value E(x) - calculated as shape/rate
        /// </summary>
        /// <returns>E(x)</returns>
        public double GetMean()
        {
            if (IsPointMass) return Point;
            double mean, variance;
            GetMeanAndVariance(out mean, out variance);
            return mean;
        }

        /// <summary>
        /// Gets the variance - calculated as shape/rate^2
        /// </summary>
        /// <returns>Variance</returns>
        public double GetVariance()
        {
            if (IsPointMass) return 0;

            double mean, variance;
            GetMeanAndVariance(out mean, out variance);
            return variance;
        }

        /// <summary>
        /// This is an asymptotic approximation to this distribution, derived from the Rocktaeschel approximation:
        /// ln Gamma(x) \approx (x-.5)*ln(x) - x +.5*ln(2*pi)
        /// </summary>
        /// <returns>A gamma approximation to this distribution</returns>
        public Gamma GammaApproximation()
        {
            return Gamma.FromShapeAndRate(Shape + .5*K*(D - 1), Rate - (K == 0.0 ? 0.0 : (K*D*Math.Log(D))));
        }

        /// <summary>
        /// Find a Laplace approximation to the density of this distribution under a change of variable x=log(y)
        /// </summary>
        /// <param name="mean">Mean of approximation</param>
        /// <param name="variance">Variance of approximation</param>
        public void SmartProposal(out double mean, out double variance)
        {
            double init = GammaApproximation().GetMeanLog();

            var mythis = this;
            Converter<double, double> f = x =>
                {
                    double expx = Math.Exp(x);
                    double res = mythis.Shape*x - mythis.Rate*expx + (mythis.K == 0.0 ? 0.0 : mythis.K*(MMath.GammaLn(mythis.D*expx) - mythis.D*MMath.GammaLn(expx)));
                    if (double.IsNaN(res))
                        throw new InferRuntimeException(errorMessage);
                    return res;
                };

            Converter<double, double> df = x =>
                {
                    double expx = Math.Exp(x);
                    double res = mythis.Shape - mythis.Rate*expx + (mythis.K == 0.0 ? 0.0 : mythis.K*mythis.D*expx*(MMath.Digamma(mythis.D*expx) - MMath.Digamma(expx)));
                    if (double.IsNaN(res))
                        throw new InferRuntimeException(errorMessage);
                    return res;
                };

            Converter<double, double> hf = x =>
                {
                    double expx = Math.Exp(x);
                    double res = -mythis.Rate*expx + (mythis.K == 0.0
                                                          ? 0.0
                                                          : mythis.K*mythis.D*expx*(MMath.Digamma(mythis.D*expx) - MMath.Digamma(expx)) +
                                                            mythis.K*mythis.D*expx*expx*(mythis.D*MMath.Trigamma(mythis.D*expx) - MMath.Trigamma(expx)));
                    if (double.IsNaN(res))
                        throw new InferRuntimeException(errorMessage);
                    return res;
                };


            mean = Newton1DMaximise(Math.Max(init, -100.0), f, df, hf);
            variance = -1.0/hf(mean);
        }

        public double GetMode()
        {
            double m, v;
            SmartProposal(out m, out v);
            return Math.Exp(m);
        }

        public static double Newton1DMaximise(double initx, Converter<double, double> f, Converter<double, double> dfxfunc, Converter<double, double> dhxfunc)
        {
            double xnew = initx;
            double x = xnew;
            int maxIters = 100; // Should only need a handful of iters
            int cnt = 0;
            do
            {
                x = xnew;
                double dir = dfxfunc(x)/dhxfunc(x);
                double step = 1.0;
                xnew = x - step*dir; // The Newton step (for maximisation)
                while (f(xnew) < f(x))
                {
                    step *= .5;
                    xnew = x - step*dir;
                    if (step == 0.0)
                        throw new InferRuntimeException(errorMessage); // could return x here?
                }
                if (double.IsNaN(xnew))
                    throw new InferRuntimeException(errorMessage);

                if (Math.Abs(x - xnew) < 0.0001)
                    break;
            } while (++cnt < maxIters);
            return xnew;
        }

        /// <summary>
        /// Approximation method to use for non-analytic expectations. 
        /// Asymptotic: use expectations under the approximating Gamma distribution
        /// GaussHermiteQuadrature: Use Gauss-Hermite quadrature with 32 quadrature points
        /// ClenshawCurtisQuadrature: Use Clenshaw Curtis quadrature with an adaptive number of quadrature points
        /// </summary>
        public enum ApproximationMethod
        {
            /// <summary>
            /// Uses the approximation Gamma(Shape+K(D-1)/2,Rate-KDlogD)
            /// </summary>
            Asymptotic,

            /// <summary>
            /// Gaussian Hermite quadrature, using Asymptotic as the proposal distribution
            /// </summary>
            GaussHermiteQuadrature,

            /// <summary>
            /// Clenshaw Curtis Quadrature
            /// </summary>
            ClenshawCurtisQuadrature,

            /// <summary>
            /// Gauss Laguerre Quadrature
            /// </summary>
            GaussLaguerreQuadrature,

            /// <summary>
            /// Gaussian Hermite quadrature, using Asymptotic as the proposal distribution if shape less than 1 and 
            /// Laplace approximation as the proposal distribution otherwise
            /// </summary>
            GaussHermiteQuadratureLaplace
        };

        /// <summary>
        /// Approximation method to use for non-analytic expectations. 
        /// </summary>
        public static ApproximationMethod approximationMethod = ApproximationMethod.GaussHermiteQuadratureLaplace;

        /// <summary>
        /// Gets the mean and variance. Note for K!=0 this requires quadrature. 
        /// </summary>
        /// <param name="mean">Where to put the mean</param>
        /// <param name="variance">Where to put the variance</param>
        public void GetMeanAndVariance(out double mean, out double variance)
        {
            // TODO: make a version which only calculates mean
            if (IsPointMass)
            {
                mean = Point;
                variance = 0;
            }
            else if (K == 0.0) // special case reduces to Gamma distribution
            {
                Gamma.FromShapeAndRate(Shape, Rate).GetMeanAndVariance(out mean, out variance);
            }
            else
            {
                double z0 = 0, z1 = 0, z2 = 0;
                double meanLog = 0, meanLog2 = 0;
                var ga = GammaApproximation();

                // Mean in the transformed domain
                double proposalMean = ga.GetMeanLog();
                // Laplace approximation of variance in transformed domain 
                double proposalVariance = 1/ga.Shape; // *10.0;
                //double proposalVariance = 1 / (ga.Rate * Math.Exp(proposalMean)); 
                int nt;
                Vector nodes, weights, p;
                double maxP;
                switch (approximationMethod)
                {
                    case ApproximationMethod.Asymptotic:
                        ga.GetMeanAndVariance(out mean, out variance);
                        return;

                    case ApproximationMethod.GaussHermiteQuadrature:
                        // Quadrature coefficient
                        nt = 50;
                        nodes = Vector.Zero(nt);
                        weights = Vector.Zero(nt);
                        Quadrature.GaussianNodesAndWeights(proposalMean, proposalVariance, nodes, weights);
                        // Precompute weights for each m slice
                        p = Vector.Zero(nt);

                        for (int i = 0; i < nt; i++)
                        {
                            double x = nodes[i];
                            var expx = Math.Exp(x);
                            p[i] = Shape*x - Rate*expx + (K == 0.0 ? 0.0 : K*(MMath.GammaLn(D*expx) - D*MMath.GammaLn(expx))) -
                                   Gaussian.GetLogProb(x, proposalMean, proposalVariance);
                            p[i] += Math.Log(weights[i]);
                        }
                        maxP = p.Max(); // this helps us avoid overflow problems
                        //double minP = p.Min();
                        for (int i = 0; i < nt; i++)
                        {
                            double x = nodes[i];
                            z0 += Math.Exp(p[i] - maxP);
                            z1 += Math.Exp(p[i] + x - maxP);
                            z2 += Math.Exp(p[i] + 2.0*x - maxP);
                            meanLog += Math.Exp(p[i] - maxP)*x;
                            meanLog2 += Math.Exp(p[i] - maxP)*x*x;
                            if (double.IsPositiveInfinity(z0))
                                throw new InferRuntimeException("Numerical problem in Conjugate Dirichlet Distribution (GetMeanAndVariance)");
                        }
                        break;

                    case ApproximationMethod.GaussHermiteQuadratureLaplace:
                        // Quadrature coefficient
                        nt = 50;
                        nodes = Vector.Zero(nt);
                        weights = Vector.Zero(nt);
                        if (ga.Shape > 1.0)
                            SmartProposal(out proposalMean, out proposalVariance);
                        Quadrature.GaussianNodesAndWeights(proposalMean, proposalVariance, nodes, weights);
                        // Precompute weights for each m slice
                        p = Vector.Zero(nt);

                        for (int i = 0; i < nt; i++)
                        {
                            double x = nodes[i];
                            var expx = Math.Exp(x);
                            p[i] = Shape*x - Rate*expx + (K == 0.0 ? 0.0 : K*(MMath.GammaLn(D*expx) - D*MMath.GammaLn(expx))) -
                                   Gaussian.GetLogProb(x, proposalMean, proposalVariance);
                            p[i] += Math.Log(weights[i]);
                        }
                        maxP = p.Max(); // this helps us avoid overflow problems
                        //double minP = p.Min();
                        for (int i = 0; i < nt; i++)
                        {
                            double x = nodes[i];
                            z0 += Math.Exp(p[i] - maxP);
                            z1 += Math.Exp(p[i] + x - maxP);
                            z2 += Math.Exp(p[i] + 2.0*x - maxP);
                            meanLog += Math.Exp(p[i] - maxP)*x;
                            meanLog2 += Math.Exp(p[i] - maxP)*x*x;
                            if (double.IsPositiveInfinity(z0))
                                throw new InferRuntimeException("Numerical problem in Conjugate Dirichlet Distribution (GetMeanAndVariance)");
                        }
                        break;

                    case ApproximationMethod.GaussLaguerreQuadrature:
                        // Quadrature coefficient
                        nt = 15;
                        nodes = Vector.Zero(nt);
                        weights = Vector.Zero(nt);
                        //Quadrature.LaguerreGammaNodesAndWeights(1, 1 /* todo shape or rate here? */, nodes, weights);

                        if (ga.Shape > 10)
                        {
                            var newprop = new ConjugateDirichlet(this);
                            newprop.Shape -= 1;
                            var mode = newprop.GetMode();
                            var h = -(Shape - 1.0)/(mode*mode) - (K == 0.0 ? 0.0 : K*(D*MMath.Trigamma(mode) - D*D*MMath.Trigamma(D*mode)));
                            ga.SetMeanAndVariance(mode, -1.0/h);
                        }

                        Quadrature.LaguerreGammaNodesAndWeights(ga.Shape, ga.Rate /* todo shape or rate here? */, nodes, weights);
                        // Precompute weights for each m slice
                        p = Vector.Zero(nt);


                        for (int i = 0; i < nt; i++)
                        {
                            double x = nodes[i];
                            p[i] = Shape*Math.Log(x) - Rate*x + (K == 0.0 ? 0.0 : K*(MMath.GammaLn(D*x) - D*MMath.GammaLn(x))) - ga.GetLogProb(x);
                            p[i] += Math.Log(weights[i]);
                        }
                        maxP = p.Max(); // this helps us avoid overflow problems
                        //double minP = p.Min();
                        for (int i = 0; i < nt; i++)
                        {
                            double x = nodes[i];
                            z0 += Math.Exp(p[i] - maxP);
                            z1 += Math.Exp(p[i] - maxP)*x;
                            z2 += Math.Exp(p[i] - maxP)*x*x;
                            if (double.IsPositiveInfinity(z0) || double.IsNaN(z0))
                                throw new InferRuntimeException("Numerical problem in Conjugate Dirichlet Distribution (GetMeanAndVariance)");
                        }
                        break;
                    case ApproximationMethod.ClenshawCurtisQuadrature:
                        // we transform the problem to be approximately centred around 0 with unit variance
                        double shape = Shape, rate = Rate, k = K, d = D;
                        double s = Math.Sqrt(proposalVariance), m = proposalMean;
                        Converter<double, double> f = x => shape*x - rate*Math.Exp(x) + (k == 0.0 ? 0.0 : k*(MMath.GammaLn(d*Math.Exp(x)) - d*MMath.GammaLn(Math.Exp(x))));
                        z0 = Quadrature.AdaptiveClenshawCurtis(x => Math.Exp(f(x*s + m)), 1.0, 64, 1e-6);
                        z1 = Quadrature.AdaptiveClenshawCurtis(x => Math.Exp(x*s + m + f(x*s + m)), 1.0, 64, 1e-6);
                        z2 = Quadrature.AdaptiveClenshawCurtis(x => Math.Exp(2.0*(x*s + m) + f(x*s + m)), 1.0, 64, 1e-6);
                        break;
                }

                mean = z1/z0;
                variance = z2/z0 - mean*mean;
                meanLog /= z0;
                meanLog2 = meanLog2/z0 - meanLog*meanLog;
                if (double.IsNaN(mean) || double.IsInfinity(mean) || double.IsNaN(variance) || double.IsInfinity(variance))
                    throw new InferRuntimeException("Numerical problem in Conjugate Dirichlet Distribution (GetMeanAndVariance)");
            }
        }

        /// <summary>
        /// Sets the mean and variance
        /// </summary>
        /// <param name="mean">Mean</param>
        /// <param name="variance">Variance</param>
        public void SetMeanAndVariance(double mean, double variance)
        {
            if (variance == 0)
            {
                Point = mean;
            }
            else if (Double.IsPositiveInfinity(variance))
            {
                SetToUniform();
            }
            else if (variance < 0)
            {
                throw new ArgumentException("variance < 0 (" + variance + ")");
            }
            else
            {
                K = 0;
                D = 0;
                Rate = mean/variance;
                Shape = mean*Rate;
            }
        }

        /// <summary>
        /// Creates a new Gamma distribution from mean and variance
        /// </summary>
        /// <param name="mean">Mean</param>
        /// <param name="variance">Variance</param>
        /// <returns>A new Gamma instance</returns>
        public static ConjugateDirichlet FromMeanAndVariance(double mean, double variance)
        {
            ConjugateDirichlet g = new ConjugateDirichlet();
            g.SetMeanAndVariance(mean, variance);
            return g;
        }

        /// <summary>
        /// Sets the shape and rate (rate = 1/scale) parameters of the distribution.
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="rate">rate = 1/scale</param>
        public void SetShapeAndRate(double shape, double rate)
        {
            this.Shape = shape;
            this.Rate = rate;
        }

        /// <summary>
        /// Constructs a Gamma distribution with the given shape and rate parameters.
        /// </summary>
        /// <param name="shape">shape</param>
        /// <param name="rate">rate = 1/scale</param>
        /// <returns>A new Gamma distribution.</returns>
        [Construction("Shape", "Rate")]
        public static ConjugateDirichlet FromShapeAndRate(double shape, double rate)
        {
            ConjugateDirichlet result = new ConjugateDirichlet();
            result.K = 0;
            result.D = 0;
            result.Shape = shape;
            result.Rate = rate;
            return result;
        }

        /// <summary>
        /// Gets the scale (1/rate)
        /// </summary>
        public double GetScale()
        {
            return 1.0/Rate;
        }

        /// <summary>
        /// Gets the shape and scale (1/rate)
        /// </summary>
        /// <param name="shape">Where to put the shape</param>
        /// <param name="scale">Where to put the scale</param>
        public void GetShapeAndScale(out double shape, out double scale)
        {
            shape = this.Shape;
            scale = 1.0/Rate;
        }

        /// <summary>
        /// Sets the shape and scale for this instance
        /// </summary>
        /// <param name="shape">Shape</param>
        /// <param name="scale">Scale</param>
        public void SetShapeAndScale(double shape, double scale)
        {
            this.Shape = shape;
            this.Rate = 1.0/scale;
        }

        /// <summary>
        /// Constructs a Gamma distribution with the given shape and scale parameters.
        /// </summary>
        /// <param name="shape">shape</param>
        /// <param name="scale">scale</param>
        /// <returns>A new Gamma distribution.</returns>
        public static ConjugateDirichlet FromShapeAndScale(double shape, double scale)
        {
            ConjugateDirichlet result = new ConjugateDirichlet();
            result.SetShapeAndScale(shape, scale);
            return result;
        }

        /// <summary>
        /// Constructs a Conjugate Dirichlet distribution with the given mean and mean logarithm.
        /// </summary>
        /// <param name="mean">Desired expected value.</param>
        /// <param name="meanLog">Desired expected logarithm.</param>
        /// <returns>A new Gamma distribution.</returns>
        /// <remarks>This function is equivalent to maximum-likelihood estimation of a Gamma distribution
        /// from data given by sufficient statistics.
        /// This function is significantly slower than the other constructors since it
        /// involves nonlinear optimization. The algorithm is a generalized Newton iteration, 
        /// described in "Estimating a Gamma distribution" by T. Minka, 2002.
        /// </remarks>
        public static ConjugateDirichlet FromMeanAndMeanLog(double mean, double meanLog)
        {
            var gamma = Gamma.FromMeanAndMeanLog(mean, meanLog);
            return ConjugateDirichlet.FromShapeAndRate(gamma.Shape, gamma.Rate);
        }

        /// <summary>
        /// Sets the natural parameters of the distribution.
        /// </summary>
        /// <param name="shapeMinus1">The shape parameter - 1.</param>
        /// <param name="rate">rate = 1/scale</param>
        /// <param name="D"></param>
        /// <param name="K"></param>
        public void SetNatural(double shapeMinus1, double rate, double D, double K)
        {
            this.D = D;
            this.K = K;
            SetShapeAndRate(shapeMinus1 + 1, rate);
        }

        /// <summary>
        /// Computes E[log(x)] using quadrature if necessary
        /// </summary>
        /// <returns>E[log(x)]</returns>
        public double GetMeanLog()
        {
            if (IsPointMass)
            {
                return Math.Log(Point);
            }
                //else if (K == 0)
                //    return Gamma.FromShapeAndRate(Shape, Rate).GetMeanLog();
            else
            {
                double z0 = 0, z1 = 0;
                var ga = GammaApproximation();

                // Mean in the transformed domain
                double proposalMean = ga.GetMeanLog();
                // Laplace approximation of variance in transformed domain 
                double proposalVariance = 1/ga.Shape;

                int nt;
                Vector nodes, weights, p;
                double maxP;

                switch (approximationMethod)
                {
                    case ApproximationMethod.Asymptotic:
                        return ga.GetMeanLog();

                    case ApproximationMethod.GaussHermiteQuadrature:
                        // Quadrature coefficient
                        nt = 32;
                        nodes = Vector.Zero(nt);
                        weights = Vector.Zero(nt);
                        Quadrature.GaussianNodesAndWeights(proposalMean, proposalVariance, nodes, weights);
                        // Precompute weights for each m slice
                        p = Vector.Zero(nt);

                        for (int i = 0; i < nt; i++)
                        {
                            double x = nodes[i];
                            var expx = Math.Exp(x);
                            p[i] = Shape*x - Rate*expx + (K == 0.0 ? 0.0 : K*(MMath.GammaLn(D*expx) - D*MMath.GammaLn(expx))) -
                                   Gaussian.GetLogProb(x, proposalMean, proposalVariance);
                            p[i] += Math.Log(weights[i]);
                        }
                        maxP = p.Max();
                        for (int i = 0; i < nt; i++)
                        {
                            double x = nodes[i];
                            z0 += Math.Exp(p[i] - maxP);
                            z1 += x*Math.Exp(p[i] - maxP);
                            if (double.IsPositiveInfinity(z0))
                                throw new InferRuntimeException("inf");
                        }
                        break;

                    case ApproximationMethod.GaussHermiteQuadratureLaplace:
                        // Quadrature coefficient
                        nt = 32;
                        nodes = Vector.Zero(nt);
                        weights = Vector.Zero(nt);
                        if (ga.Shape > 1.0)
                            SmartProposal(out proposalMean, out proposalVariance);
                        Quadrature.GaussianNodesAndWeights(proposalMean, proposalVariance, nodes, weights);
                        // Precompute weights for each m slice
                        p = Vector.Zero(nt);

                        for (int i = 0; i < nt; i++)
                        {
                            double x = nodes[i];
                            var expx = Math.Exp(x);
                            p[i] = Shape*x - Rate*expx + (K == 0.0 ? 0.0 : K*(MMath.GammaLn(D*expx) - D*MMath.GammaLn(expx))) -
                                   Gaussian.GetLogProb(x, proposalMean, proposalVariance);
                            p[i] += Math.Log(weights[i]);
                        }
                        maxP = p.Max();
                        for (int i = 0; i < nt; i++)
                        {
                            double x = nodes[i];
                            z0 += Math.Exp(p[i] - maxP);
                            z1 += x*Math.Exp(p[i] - maxP);
                            if (double.IsPositiveInfinity(z0))
                                throw new InferRuntimeException("inf");
                        }
                        break;

                    case ApproximationMethod.ClenshawCurtisQuadrature:

                        double shape = Shape, rate = Rate, k = K, d = D;
                        double s = Math.Sqrt(proposalVariance), m = proposalMean;
                        Converter<double, double> f = x => shape*x - rate*Math.Exp(x)
                                                           + (k == 0.0 ? 0.0 : k*(MMath.GammaLn(d*Math.Exp(x)) - d*MMath.GammaLn(Math.Exp(x))));
                        z0 = Quadrature.AdaptiveClenshawCurtis(x => Math.Exp(f(x*s + m)), 1.0, 64, 1e-6);
                        z1 = Quadrature.AdaptiveClenshawCurtis(x => (x*s + m)*Math.Exp(f(x*s + m)), 1.0, 64, 1e-6);
                        break;
                }

                double res = z1/z0;
                if (double.IsNaN(res) || double.IsInfinity(res))
                    throw new InferRuntimeException("Numerical problem in Conjugate Dirichlet Distribution (GetMeanLog)");
                return res;
            }
        }

        private double GammaLnExp(double x)
        {
            if (x < -100.0)
                return -x;
            else
                return MMath.GammaLn(Math.Exp(x));
        }

        /// <summary>
        ///  Compute E[log G(factor * x)] using quadrature
        /// </summary>
        /// <param name="factor">Multiplier for x</param>
        /// <returns>E[log G(factor * x)]</returns>
        public double GetMeanLogGamma(double factor)
        {
            if (IsPointMass)
            {
                return Math.Log(Point);
            }
            else
            {
                double z0 = 0, z1 = 0;
                var ga = GammaApproximation();

                // Mean in the transformed domain
                double proposalMean = ga.GetMeanLog();
                // Laplace approximation of variance in transformed domain 
                double proposalVariance = 1/ga.Shape;
                int nt;
                Vector nodes, weights, p;
                double maxP;
                switch (approximationMethod)
                {
                    case ApproximationMethod.Asymptotic:
                        return GammaFromShapeAndRateOp.ELogGamma(Gamma.FromShapeAndRate(ga.Shape, ga.Rate/factor));

                    case ApproximationMethod.GaussHermiteQuadrature:
                        // Quadrature coefficient
                        nt = 32;
                        nodes = Vector.Zero(nt);
                        weights = Vector.Zero(nt);
                        Quadrature.GaussianNodesAndWeights(proposalMean, proposalVariance, nodes, weights);
                        // Precompute weights for each m slice
                        p = Vector.Zero(nt);

                        for (int i = 0; i < nt; i++)
                        {
                            double x = nodes[i];
                            var expx = Math.Exp(x);
                            p[i] = Shape*x - Rate*expx + (K == 0.0 ? 0.0 : K*(GammaLnExp(Math.Log(D) + x) - D*GammaLnExp(x))) -
                                   Gaussian.GetLogProb(x, proposalMean, proposalVariance);
                            p[i] += Math.Log(weights[i]);
                        }
                        maxP = p.Max();
                        for (int i = 0; i < nt; i++)
                        {
                            double x = nodes[i];
                            z0 += Math.Exp(p[i] - maxP);
                            z1 += GammaLnExp(Math.Log(factor) + x)*Math.Exp(p[i] - maxP);
                            if (double.IsInfinity(z0) || double.IsInfinity(z1))
                                throw new InferRuntimeException("inf");
                        }
                        break;
                    case ApproximationMethod.GaussHermiteQuadratureLaplace:
                        // Quadrature coefficient
                        nt = 32;
                        nodes = Vector.Zero(nt);
                        weights = Vector.Zero(nt);
                        if (ga.Shape > 1.0)
                            SmartProposal(out proposalMean, out proposalVariance);
                        Quadrature.GaussianNodesAndWeights(proposalMean, proposalVariance, nodes, weights);
                        // Precompute weights for each m slice
                        p = Vector.Zero(nt);

                        for (int i = 0; i < nt; i++)
                        {
                            double x = nodes[i];
                            var expx = Math.Exp(x);
                            p[i] = Shape*x - Rate*expx + (K == 0.0 ? 0.0 : K*(GammaLnExp(Math.Log(D) + x) - D*GammaLnExp(x))) -
                                   Gaussian.GetLogProb(x, proposalMean, proposalVariance);
                            p[i] += Math.Log(weights[i]);
                        }
                        maxP = p.Max();
                        for (int i = 0; i < nt; i++)
                        {
                            double x = nodes[i];
                            z0 += Math.Exp(p[i] - maxP);
                            z1 += GammaLnExp(Math.Log(factor) + x)*Math.Exp(p[i] - maxP);
                            if (double.IsPositiveInfinity(z0))
                                throw new InferRuntimeException("inf");
                        }
                        break;
                    case ApproximationMethod.ClenshawCurtisQuadrature:

                        double shape = Shape, rate = Rate, k = K, d = D;
                        double s = Math.Sqrt(proposalVariance), m = proposalMean;
                        Converter<double, double> f = x => shape*x - rate*Math.Exp(x) + (k == 0.0 ? 0.0 : k*(MMath.GammaLn(d*Math.Exp(x)) - d*MMath.GammaLn(Math.Exp(x))));
                        z0 = Quadrature.AdaptiveClenshawCurtis(x => Math.Exp(f(x*s + m)), 1.0, 64, 1e-6);
                        z1 = Quadrature.AdaptiveClenshawCurtis(x => MMath.GammaLn(factor*Math.Exp(x*s + m))*Math.Exp(f(x*s + m)), 1.0, 64, 1e-6);
                        break;
                }
                double res = z1/z0;
                if (double.IsNaN(res) || double.IsInfinity(res))
                    throw new InferRuntimeException("Numerical problem in Conjugate Dirichlet Distribution (GetMeanLogGamma)");
                return res;
            }
        }

        /// <summary>
        /// Computes E[1/x]
        /// </summary>
        /// <returns></returns>
        public double GetMeanInverse()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Computes E[x^power]
        /// </summary>
        /// <returns></returns>
        public double GetMeanPower(double power)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Asks whether the instance is a point mass
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public bool IsPointMass
        {
            get { return (Shape == Double.PositiveInfinity); }
        }

        /// <summary>
        /// Sets this instance to a point mass. The location of the
        /// point mass is the existing Rate parameter
        /// </summary>
        private void SetToPointMass()
        {
            Shape = Double.PositiveInfinity;
        }

        /// <summary>
        /// Sets/gets the instance as a point mass
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public double Point
        {
            get
            {
                // The accessor must succeed, even if the distribution is not a point mass.
                //Assert.IsTrue(IsPointMass, "The distribution is not a point mass");
                return Rate;
            }
            set
            {
                SetToPointMass();
                Rate = value;
            }
        }

        /// <summary>
        /// Sets this Conjugate Dirichlet instance to be a uniform distribution
        /// </summary>
        public void SetToUniform()
        {
            K = 0;
            D = 0;
            Shape = 1;
            Rate = 0;
        }

        /// <summary>
        /// Asks whether this instance is uniform
        /// </summary>
        /// <returns>True if uniform, false otherwise</returns>
        public bool IsUniform()
        {
            return (Shape == 1) && (Rate == 0) && (K == 0);
        }

        /// <summary>
        /// Logarithm of the Conjugate Dirichlet density function.
        /// </summary>
        /// <param name="x">Where to evaluate the density</param>
        /// <param name="shape">Shape parameter</param>
        /// <param name="rate">Rate parameter</param>
        /// <param name="D">D parameter</param>
        /// <param name="K">K parameter</param>
        /// <returns>log(Conjugate Dirichlet(x;shape,rate,D,K))</returns>
        /// <remarks>
        /// The distribution is <c>x^{Shape-1} exp(-Rate*x) / B(x,D)^K
        /// where B(x,D)=Gamma(x)^D/Gamma(D*x)</c>
        /// </remarks>
        public static double GetLogProb(double x, double shape, double rate, double D, double K)
        {
            return ConjugateDirichlet.FromNatural(shape - 1, rate, D, K).GetLogProb(x);
        }

        /// <summary>
        /// Logarithm of this Conjugate Dirichlet density function.
        /// </summary>
        /// <param name="x">Where to evaluate the density</param>
        /// <returns>log(Conjugate Dirichlet(x))</returns>
        public double GetLogProb(double x)
        {
            if (IsPointMass)
            {
                return (x == Point) ? 0.0 : Double.NegativeInfinity;
            }
            else if (IsUniform())
            {
                return 0;
            }
            else
            {
                return GetUnnormalisedLogProb(x) - GetLogNormalizer();
            }
        }

        // x^{Shape-1} exp(-Rate*x) / B(x,D)^K
        // where B(x,D)=Gamma(x)^D/Gamma(D*x)
        public double GetUnnormalisedLogProb(double x)
        {
            if (IsPointMass)
            {
                return (x == Point) ? 0.0 : Double.NegativeInfinity;
            }
            else if (IsUniform())
            {
                return 0;
            }
            else
            {
                if (x < 0)
                    return double.NegativeInfinity;
                else
                {
                    double result = (Shape - 1.0)*Math.Log(x) - Rate*x;
                    if (K != 0.0)
                        result += K*(MMath.GammaLn(D*x) - D*MMath.GammaLn(x));
                    return result;
                }
            }
        }

        /// <summary>
        /// Gets log normalizer
        /// </summary>
        /// <returns></returns>
        public double GetLogNormalizer()
        {
            if (IsProper())
            {
                var ga = GammaApproximation();
                // Mean in the transformed domain
                double proposalMean = ga.GetMeanLog();
                // Laplace approximation of variance in transformed domain 
                double proposalVariance = 1/ga.Shape;
                // Quadrature coefficient
                int nt = 32;
                Vector nodes = Vector.Zero(nt);
                Vector weights = Vector.Zero(nt);
                Quadrature.GaussianNodesAndWeights(proposalMean, proposalVariance, nodes, weights);
                // Precompute weights for each m slice
                var p = new double[nt];
                for (int i = 0; i < nt; i++)
                {
                    double x = nodes[i];
                    var expx = Math.Exp(x);
                    p[i] = Shape*x - Rate*expx + (K == 0.0 ? 0.0 : K*(MMath.GammaLn(D*expx) - D*MMath.GammaLn(expx)))
                           - Gaussian.GetLogProb(x, proposalMean, proposalVariance)
                           + Math.Log(weights[i]);
                }
                double res = MMath.LogSumExp(p);
                if (double.IsNaN(res) || double.IsInfinity(res))
                    throw new InferRuntimeException("Numerical problem in Conjugate Dirichlet Distribution (GetLogNormalizer)");
                return res;
            }
            else
            {
                return 0.0;
            }
        }

        /// <summary>
        /// The expected logarithm of that distribution under this distribution.
        /// </summary>
        /// <param name="that">The distribution to take the logarithm of.</param>
        /// <returns><c>sum_x this.Evaluate(x)*Math.Log(that.Evaluate(x))</c></returns>
        /// <remarks>This is also known as the cross entropy.</remarks>
        public double GetAverageLog(ConjugateDirichlet that)
        {
            if (that.IsPointMass)
            {
                if (this.IsPointMass && this.Point.Equals(that.Point)) return 0.0;
                else return Double.NegativeInfinity;
            }
            else if (!IsProper())
            {
                throw new ImproperDistributionException(this);
            }
            else
            {
                // that is not a point mass.
                double result = (that.Shape - 1)*GetMeanLog() - GetMean()*that.Rate;
                if (that.K > 0.0)
                    result -= that.K*(that.D*GetMeanLogGamma(1.0) - GetMeanLogGamma(that.D));
                result -= that.GetLogNormalizer();
                if (double.IsInfinity(result) || double.IsNaN(result))
                    throw new InferRuntimeException("Numerical problem in Conjugate Dirichlet Distribution (GetAverageLog)");
                return result;
            }
        }

        /// <summary>
        /// Asks whether this Conjugate Dirichlet instance is proper or not. A Conjugate Dirichlet distribution
        /// is proper if it's approximating gamma is proper
        /// </summary>
        /// <returns>True if proper, false otherwise</returns>
        public bool IsProper()
        {
            return GammaApproximation().IsProper();
        }

        /// <summary>
        /// Asks whether a Conjugate Dirichlet distribution is proper or not. A Conjugate Dirichlet distribution
        /// is proper only if Shape > 0 and Rate > 0.
        /// </summary>
        /// <param name="shape">shape parameter for the Conjugate Dirichlet</param>
        /// <param name="rate">rate parameter for the Conjugate Dirichlet</param>
        /// <returns>True if proper, false otherwise</returns>
        public static bool IsProper(double shape, double rate)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// The log of the integral of the product of this Conjugate Dirichlet and that Conjugate Dirichlet
        /// </summary>
        /// <param name="that">That Conjugate Dirichlet</param>
        /// <returns>The log inner product</returns>
        public double GetLogAverageOf(ConjugateDirichlet that)
        {
            if (IsPointMass)
            {
                return that.GetLogProb(Point);
            }
            else if (that.IsPointMass)
            {
                return GetLogProb(that.Point);
            }
            else
            {
                var product = this*that;
                //if (!product.IsProper()) throw new ArgumentException("The product is improper.");
                return product.GetLogNormalizer() - this.GetLogNormalizer() - that.GetLogNormalizer();
            }
        }

        /// <summary>
        /// Samples from this Conjugate Dirichlet distribution
        /// </summary>
        /// <returns>The sample value</returns>
        [Stochastic]
        public double Sample()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Samples from this Conjugate Dirichlet distribution
        /// </summary>
        /// <param name="result">Ignored</param>
        /// <returns>The sample value</returns>
        [Stochastic]
        public double Sample(double result)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Samples from a Conjugate Dirichlet distribution with given shape and scale
        /// </summary>
        /// <param name="shape">shape parameter</param>
        /// <param name="scale">scale parameter</param>
        /// <returns>The sample value</returns>
        [Stochastic]
        public static double Sample(double shape, double scale)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Samples from a Conjugate Dirichlet distribution with given mean and variance
        /// </summary>
        /// <param name="mean">mean parameter</param>
        /// <param name="variance">variance parameter</param>
        /// <returns>The sample value</returns>
        [Stochastic]
        [ParameterNames("sample", "mean", "variance")]
        public static double SampleFromMeanAndVariance(double mean, double variance)
        {
            return FromMeanAndVariance(mean, variance).Sample();
        }

        /// <summary>
        /// Sets this Conjugate Dirichlet instance to have the parameter values of that Conjugate Dirichlet instance
        /// </summary>
        /// <param name="that">That Conjugate Dirichlet</param>
        public void SetTo(ConjugateDirichlet that)
        {
            K = that.K;
            D = that.D;
            Rate = that.Rate;
            Shape = that.Shape;
        }

        /// <summary>
        /// Sets the parameters to represent the product of two Gammas.
        /// </summary>
        /// <param name="a">The first Conjugate Dirichlet</param>
        /// <param name="b">The second Conjugate Dirichlet</param>
        /// <remarks>
        /// The result may not be proper. No error is thrown in this case.
        /// </remarks>
        public void SetToProduct(ConjugateDirichlet a, ConjugateDirichlet b)
        {
            if (a.IsPointMass)
            {
                if (b.IsPointMass && !a.Point.Equals(b.Point))
                {
                    throw new AllZeroException();
                }
                Point = a.Point;
                return;
            }
            else if (b.IsPointMass)
            {
                Point = b.Point;
                return;
            }
            else
            {
                Shape = a.Shape + b.Shape - 1;
                Rate = a.Rate + b.Rate;
                K = a.K + b.K;
                D = a.D;
                if (a.K == 0)
                    D = b.D;
                else if (b.K == 0)
                    D = a.D;
                else if (a.D != b.D)
                    throw new InferRuntimeException(
                        "Cannot multiple ConjugateDirichlet distributions with different D, i.e. you cannot have the same hyperparameter for two Dirichlet distributions with different dimensions");
            }
        }

        /// <summary>
        /// Creates a new Conjugate Dirichlet which the product of two other Conjugate Dirichlets
        /// </summary>
        /// <param name="a">First Conjugate Dirichlet</param>
        /// <param name="b">Second Conjugate Dirichlet</param>
        /// <returns>Result</returns>
        public static ConjugateDirichlet operator *(ConjugateDirichlet a, ConjugateDirichlet b)
        {
            ConjugateDirichlet result = new ConjugateDirichlet();
            result.SetToProduct(a, b);
            return result;
        }

        /// <summary>
        /// Sets the parameters to represent the ratio of two Conjugate Dirichlets.
        /// </summary>
        /// <param name="numerator">The numerator Conjugate Dirichlet</param>
        /// <param name="denominator">The denominator Conjugate Dirichlet</param>
        /// <param name="forceProper"></param>
        /// <remarks>
        /// The result may not be proper. No error is thrown in this case.
        /// </remarks>
        public void SetToRatio(ConjugateDirichlet numerator, ConjugateDirichlet denominator, bool forceProper)
        {
            if (numerator.IsPointMass)
            {
                if (denominator.IsPointMass)
                {
                    if (numerator.Point.Equals(denominator.Point))
                    {
                        SetToUniform();
                    }
                    else
                    {
                        throw new DivideByZeroException();
                    }
                }
                else
                {
                    Point = numerator.Point;
                }
            }
            else if (denominator.IsPointMass)
            {
                throw new DivideByZeroException();
            }
            else
            {
                Shape = numerator.Shape - denominator.Shape + 1;
                Rate = numerator.Rate - denominator.Rate;
                K = numerator.K - denominator.K;
                D = numerator.D;
                if (numerator.K == 0)
                    D = denominator.D;
                else if (denominator.K == 0)
                    D = numerator.D;
                else if (numerator.D != denominator.D)
                    throw new InferRuntimeException(
                        "Cannot divide ConjugateDirichlet distributions with different D, i.e. you cannot have the same hyperparameter for two Dirichlet distributions with different dimensions");
            }
        }

        /// <summary>
        /// Creates a new Conjugate Dirichlet which the ratio of two other Conjugate Dirichlets
        /// </summary>
        /// <param name="numerator">numerator Conjugate Dirichlet</param>
        /// <param name="denominator">denominator Conjugate Dirichlet</param>
        /// <returns>Result</returns>
        public static ConjugateDirichlet operator /(ConjugateDirichlet numerator, ConjugateDirichlet denominator)
        {
            ConjugateDirichlet result = new ConjugateDirichlet();
            result.SetToRatio(numerator, denominator, false);
            return result;
        }

        /// <summary>
        /// Sets the parameters to represent the power of a source Conjugate Dirichlet to some exponent.
        /// </summary>
        /// <param name="dist">The source Conjugate Dirichlet</param>
        /// <param name="exponent">The exponent</param>
        public void SetToPower(ConjugateDirichlet dist, double exponent)
        {
            if (dist.IsPointMass)
            {
                if (exponent == 0)
                {
                    SetToUniform();
                }
                else if (exponent < 0)
                {
                    throw new DivideByZeroException("The exponent is negative and the distribution is a point mass");
                }
                else
                {
                    Point = dist.Point;
                }
                return;
            }
            else
            {
                Shape = (dist.Shape - 1)*exponent + 1;
                Rate = dist.Rate*exponent;
                K = dist.K*exponent;
                D = dist.D;
            }
        }

        /// <summary>
        /// Raises a distribution to a power.
        /// </summary>
        /// <param name="dist">The distribution.</param>
        /// <param name="exponent">The power to raise to.</param>
        /// <returns><paramref name="dist"/> raised to power <paramref name="exponent"/>.</returns>
        public static ConjugateDirichlet operator ^(ConjugateDirichlet dist, double exponent)
        {
            ConjugateDirichlet result = new ConjugateDirichlet();
            result.SetToPower(dist, exponent);
            return result;
        }

        /// <summary>
        /// Set the mean and variance to match a mixture of two Gammas.
        /// </summary>
        /// <param name="weight1">The first weight</param>
        /// <param name="dist1">The first Conjugate Dirichlet</param>
        /// <param name="weight2">The second weight</param>
        /// <param name="dist2">The second Conjugate Dirichlet</param>
        public void SetToSum(double weight1, ConjugateDirichlet dist1, double weight2, ConjugateDirichlet dist2)
        {
            SetTo(Gaussian.WeightedSum<ConjugateDirichlet>(this, weight1, dist1, weight2, dist2));
        }

        /// <summary>
        /// The maximum difference between the parameters of this Conjugate Dirichlet
        /// and that Conjugate Dirichlet
        /// </summary>
        /// <param name="thatd">That Conjugate Dirichlet</param>
        /// <returns>The maximum difference</returns>
        /// <remarks><c>a.MaxDiff(b) == b.MaxDiff(a)</c></remarks>
        public double MaxDiff(object thatd)
        {
            if (!(thatd is ConjugateDirichlet)) return Double.PositiveInfinity;
            ConjugateDirichlet that = (ConjugateDirichlet) thatd;
            return Math.Max(Math.Max(MMath.AbsDiff(Rate, that.Rate), MMath.AbsDiff(Shape, that.Shape)), MMath.AbsDiff(K, that.K));
        }

        /// <summary>
        /// Override of the Equals method
        /// </summary>
        /// <param name="thatd">The instance to compare to</param>
        /// <returns>True if the two distributions are the same in value, false otherwise</returns>
        /// Array of distribution requiring the distribution type to be a value type.
        /// <exclude/>
        public override bool Equals(object thatd)
        {
            return (MaxDiff(thatd) == 0.0);
        }

        /// <summary>
        /// Override of GetHashCode method
        /// </summary>
        /// <returns>The hash code for this instance</returns>
        /// <exclude/>
        public override int GetHashCode()
        {
            return Hash.Combine(Rate.GetHashCode(), Shape.GetHashCode());
        }

        /// <summary>
        /// Equals operator
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static bool operator ==(ConjugateDirichlet a, ConjugateDirichlet b)
        {
            return a.Equals(b);
        }

        /// <summary>
        /// Not equals operator
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static bool operator !=(ConjugateDirichlet a, ConjugateDirichlet b)
        {
            return !(a == b);
        }

#if class
        public ConjugateDirichlet()
        {
            SetToUniform();
        }
#endif

        /// <summary>
        /// Creates a Conjugate Dirichlet distribution with given shape and scale parameters (scale = 1/rate) 
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="scale">scale = 1/rate</param>
        /// <param name="D"></param>
        /// <param name="K"></param>
        /// <remarks>
        /// The distribution is <c>x^{Shape-1} exp(-Rate*x) / B(x,D)^K
        /// where B(x,D)=Gamma(x)^D/Gamma(D*x)</c>.
        /// </remarks>
        public ConjugateDirichlet(double shape, double scale, double D, double K)
        {
            Shape = shape;
            Rate = 1.0/scale;
            this.D = D;
            this.K = K;
        }

        /// <summary>
        /// Constructs a Conjugate Dirichlet distribution from its natural parameters.
        /// </summary>
        /// <param name="shapeMinus1">shape - 1</param>
        /// <param name="rate">rate = 1/scale</param>
        /// <param name="D"></param>
        /// <param name="K"></param>
        /// <returns>A new Conjugate Dirichlet distribution</returns>
        public static ConjugateDirichlet FromNatural(double shapeMinus1, double rate, double D, double K)
        {
            ConjugateDirichlet result = new ConjugateDirichlet();

            result.SetShapeAndRate(shapeMinus1 + 1, rate);
            result.D = D;
            result.K = K;
            return result;
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that"></param>
        public ConjugateDirichlet(ConjugateDirichlet that)
        {
            Shape = that.Shape;
            Rate = that.Rate;
            D = that.D;
            K = that.K;
            //SetTo(that);
        }

        /// <summary>
        /// Clones this Conjugate Dirichlet. 
        /// </summary>
        /// <returns>An object which is a clone of the current instance. This must be cast
        /// if you want to assign the result to a Conjugate Dirichlet type</returns>
        public object Clone()
        {
            return new ConjugateDirichlet(this);
        }

        /// <summary>
        /// Create a uniform Conjugate Dirichlet distribution.
        /// </summary>
        /// <returns>A new uniform Conjugate Dirichlet distribution</returns>
        [Construction(UseWhen = "IsUniform"), Skip]
        public static ConjugateDirichlet Uniform()
        {
            ConjugateDirichlet result = new ConjugateDirichlet();
            result.SetToUniform();
            return result;
        }

        /// <summary>
        /// Creates a point mass Conjugate Dirichlet distribution
        /// </summary>
        /// <param name="mean">The location of the point mass</param>
        /// <returns>A new point mass Conjugate Dirichlet distribution</returns>
        public static ConjugateDirichlet PointMass(double mean)
        {
            ConjugateDirichlet result = new ConjugateDirichlet();
            result.Point = mean;
            return result;
        }

        /// <summary>
        /// ToString override
        /// </summary>
        /// <returns>String representation of the instance</returns>
        /// <exclude/>
        public override string ToString()
        {
            if (IsPointMass)
            {
                return "ConjugateDirichlet.PointMass(" + Point.ToString("g4") + ")";
            }
            else if (IsUniform())
            {
                return "ConjugateDirichlet.Uniform";
            }
            else
            {
                double shape, scale;
                GetShapeAndScale(out shape, out scale);
                string s = "ConjugateDirichlet(" + shape.ToString("g4") + ", " + scale.ToString("g4") + ", D=" + D.ToString("g4") + ", K=" + K.ToString("g4") + ")";
                if (IsProper())
                {
                    s += "[mean=" + GetMean().ToString("g4") + "]";
                }
                return s;
            }
        }

        public double GetLogAverageOfPower(ConjugateDirichlet that, double power)
        {
            throw new NotImplementedException();
        }
    }
}