// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp"]/doc/*'/>
    [FactorMethod(typeof(Math), "Exp", typeof(double), Default = true)]
    [Quality(QualityBand.Stable)]
    public class ExpOp
    {
        /// <summary>
        /// The number of quadrature nodes used to compute the messages.
        /// Reduce this number to save time in exchange for less accuracy.
        /// </summary>
        public static int QuadratureNodeCount = 21;

        /// <summary>
        /// Number of quadrature iterations
        /// </summary>
        public static int QuadratureIterations = 2;

        /// <summary>
        /// Quadrature shift
        /// </summary>
        public static bool QuadratureShift = false; // gives a bit more accuracy when to_d is uniform.

        /// <summary>
        ///  Forces proper messages when set to true. 
        /// </summary>
        public static bool ForceProper = true;

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp"]/message_doc[@name="LogAverageFactor(double, double)"]/*'/>
        public static double LogAverageFactor(double exp, double d)
        {
            return (exp == Math.Exp(d)) ? 0.0 : Double.NegativeInfinity;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp"]/message_doc[@name="LogEvidenceRatio(double, double)"]/*'/>
        public static double LogEvidenceRatio(double exp, double d)
        {
            return LogAverageFactor(exp, d);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp"]/message_doc[@name="AverageLogFactor(double, double)"]/*'/>
        public static double AverageLogFactor(double exp, double d)
        {
            return LogAverageFactor(exp, d);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp"]/message_doc[@name="LogAverageFactor(CanGetLogProb{double}, double)"]/*'/>
        public static double LogAverageFactor(CanGetLogProb<double> exp, double d)
        {
            return exp.GetLogProb(Math.Exp(d));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp"]/message_doc[@name="LogAverageFactor(double, Gaussian)"]/*'/>
        public static double LogAverageFactor(double exp, Gaussian d)
        {
            double log = Math.Log(exp);
            return d.GetLogProb(log) - log;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp"]/message_doc[@name="LogAverageFactor(Gamma, Gaussian, Gaussian)"]/*'/>
        public static double LogAverageFactor(Gamma exp, Gaussian d, Gaussian to_d)
        {
            if (d.IsPointMass)
                return LogAverageFactor(exp, d.Point);
            if (d.IsUniform())
                return exp.GetLogAverageOf(new Gamma(0, 0));
            if (exp.IsPointMass)
                return LogAverageFactor(exp.Point, d);
            if (exp.IsUniform())
                return 0.0;
            double[] nodes = new double[QuadratureNodeCount];
            double[] weights = new double[QuadratureNodeCount];
            Gaussian dMarginal = d * to_d;
            dMarginal.GetMeanAndVariance(out double mD, out double vD);
            Quadrature.GaussianNodesAndWeights(mD, vD, nodes, weights);
            if (!to_d.IsUniform())
            {
                // modify the weights to include q(y_k)/N(y_k;m,v)
                for (int i = 0; i < weights.Length; i++)
                {
                    weights[i] *= Math.Exp(d.GetLogProb(nodes[i]) - Gaussian.GetLogProb(nodes[i], mD, vD));
                }
            }
            double Z = 0;
            for (int i = 0; i < weights.Length; i++)
            {
                double y = nodes[i];
                double f = weights[i] * Math.Exp((exp.Shape - 1) * y - exp.Rate * Math.Exp(y));
                Z += f;
            }
            return Math.Log(Z) - exp.GetLogNormalizer();
        }

        /// <summary>
        /// Evidence message for EP
        /// </summary>
        /// <param name="exp">Incoming message from 'exp'.</param>
        /// <param name="d">Incoming message from 'd'.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions</returns>
        /// <remarks><para>
        /// The formula for the result is <c>log(sum_(exp,d) p(exp,d) factor(exp,d))</c>.
        /// </para></remarks>
        public static double LogAverageFactor_slow(Gamma exp, Gaussian d)
        {
            Gaussian to_d = Gaussian.Uniform();
            for (int i = 0; i < QuadratureIterations; i++)
            {
                to_d = DAverageConditional(exp, d, to_d);
            }
            return LogAverageFactor(exp, d, to_d);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp"]/message_doc[@name="LogEvidenceRatio(Gamma, Gaussian, Gamma, Gaussian)"]/*'/>
        public static double LogEvidenceRatio(Gamma exp, Gaussian d, [Fresh] Gamma to_exp, Gaussian to_d)
        {
            //Gaussian to_d = Gaussian.Uniform();
            //for (int i = 0; i < QuadratureIterations; i++) {
            //  to_d = DAverageConditional(exp, d, to_d);
            //}
            //Gamma to_exp = ExpAverageConditional(exp, d, to_d);
            return LogAverageFactor(exp, d, to_d)
                                - to_exp.GetLogAverageOf(exp);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp"]/message_doc[@name="LogEvidenceRatio(double, Gaussian)"]/*'/>
        public static double LogEvidenceRatio(double exp, Gaussian d)
        {
            return LogAverageFactor(exp, d);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp"]/message_doc[@name="LogEvidenceRatio(GammaPower, Gaussian, GammaPower, Gaussian)"]/*'/>
        public static double LogEvidenceRatio(GammaPower exp, Gaussian d, [Fresh] GammaPower to_exp, Gaussian to_d)
        {
            return LogAverageFactor(exp, d, to_d)
                                - to_exp.GetLogAverageOf(exp);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp"]/message_doc[@name="LogAverageFactor(GammaPower, Gaussian, Gaussian)"]/*'/>
        public static double LogAverageFactor(GammaPower exp, Gaussian d, Gaussian to_d)
        {
            if (d.IsPointMass)
                return LogAverageFactor(exp, d.Point);
            if (d.IsUniform())
                return exp.GetLogAverageOf(new GammaPower(0, 0, exp.Power));
            if (exp.IsPointMass)
                return LogAverageFactor(exp.Point, d);
            if (exp.IsUniform())
                return 0.0;
            double oneMinusPower = 1 - exp.Power;
            bool useMethod1 = false;
            if (useMethod1)
            {
                // The factor is Ga(exp(d); shape, rate, power)
                // = exp(d*(shape/power - 1) - rate*exp(d/power))*rate^shape/Gamma(shape)/abs(power)
                // = Ga(exp(d/power); shape-power+1, rate)*Gamma(shape-power+1)/rate^(shape-power+1)*rate^shape/Gamma(shape)/abs(power)
                Gamma gamma = Gamma.FromShapeAndRate(exp.Shape + oneMinusPower, exp.Rate);
                double correction = oneMinusPower * (MMath.RisingFactorialLnOverN(exp.Shape, oneMinusPower) + Math.Log(exp.Rate)) - Math.Log(Math.Abs(exp.Power));
                Gaussian forward = GaussianProductOp.AAverageConditional(d, exp.Power);
                Gaussian backward = GaussianProductOp.ProductAverageConditional(to_d, exp.Power);
                return LogAverageFactor(gamma, forward, backward) + correction;
            }
            else
            {
                // The GammaPower density p_exp(y) = p_g(y^(1/c)) y^(1/c - 1)/abs(c)
                // int p_exp(exp(x)) p_x(x) dx 
                // = int p_g(exp(x/c)) exp(x(1/c - 1))/abs(c) p_x(x) dx
                // = int p_g(exp(z)) exp(z(1 - c)) p_x(zc) dz
                // where z = x/c
                // p_x(zc) = N(z; m_x/c, v_x/c^2)/abs(c)
                // exp(z(1-c)) p_x(zc) = exp((1-c)m_x/c + (1-c)^2 v_x/c^2/2) N(z; m_x/c + (1-c)v_x/c^2, v_x/c^2)
                // p_g(exp(z)) exp(z(1 - c)) = p_g(exp(z); shape+1-c, rate)*Gamma(shape+1-c)/rate^(shape+1-c)*rate^shape/Gamma(shape)
                Gamma gamma = Gamma.FromShapeAndRate(exp.Shape, exp.Rate);
                d.GetMeanAndVariance(out double mD, out double vD);
                Gaussian dOverPower = GaussianProductOp.AAverageConditional(d, exp.Power);
                dOverPower.SetMeanAndPrecision(dOverPower.GetMean() + oneMinusPower / dOverPower.Precision, dOverPower.Precision);
                Gaussian to_dOverPower = GaussianProductOp.ProductAverageConditional(to_d, exp.Power);
                double oneMinusPowerOverPower = oneMinusPower / exp.Power;
                double correction = oneMinusPowerOverPower * mD + oneMinusPowerOverPower * oneMinusPowerOverPower * vD / 2 - Math.Log(Math.Abs(exp.Power));
                //double correction = MMath.GammaLn(exp.Shape - exp.Power + 1) - MMath.GammaLn(exp.Shape) + (1 - exp.Power) * Math.Log(exp.Rate) - Math.Log(Math.Abs(exp.Power));
                return LogAverageFactor(gamma, dOverPower, to_dOverPower) + correction;
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp"]/message_doc[@name="ExpAverageConditional(GammaPower, Gaussian, Gaussian)"]/*'/>
        public static GammaPower ExpAverageConditional([NoInit] GammaPower exp, [Proper] Gaussian d, [NoInit] Gaussian to_d)
        {
            if (exp.IsUniform()) return ExpAverageLogarithm(d, exp);
            Gamma gamma = Gamma.FromShapeAndRate(exp.Shape + (1 - exp.Power), exp.Rate);
            if (exp.IsPointMass) gamma = Gamma.PointMass(Math.Pow(exp.Point, 1 / exp.Power));
            Gaussian dOverPower = GaussianProductOp.AAverageConditional(d, exp.Power);
            Gaussian to_dOverPower = GaussianProductOp.AAverageConditional(to_d, exp.Power);
            Gamma to_gamma = ExpAverageConditional(gamma, dOverPower, to_dOverPower);
            return PowerOp.PowAverageConditional(to_gamma, exp.Power);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp"]/message_doc[@name="DAverageConditional(GammaPower, Gaussian)"]/*'/>
        public static Gaussian DAverageConditional([SkipIfUniform] GammaPower exp, [Proper] Gaussian d, [NoInit] Gaussian to_d)
        {
            if (exp.IsPointMass) return DAverageConditional(exp.Point);
            // as a function of d, the factor is Ga(exp(d); shape, rate, power) = exp(d*(shape/power -1) -rate*exp(d/power))
            // = Ga(exp(d/power); shape - power + 1, rate)
            Gamma gamma = Gamma.FromShapeAndRate(exp.Shape + (1 - exp.Power), exp.Rate);
            // "forward" goes to d/power
            Gaussian forward = GaussianProductOp.AAverageConditional(d, exp.Power);
            Gaussian to_dOverPower = GaussianProductOp.AAverageConditional(to_d, exp.Power);
            Gaussian message = DAverageConditional(gamma, forward, to_dOverPower);
            // "message" goes to d/power
            Gaussian backward = GaussianProductOp.ProductAverageConditional(message, exp.Power);
            return backward;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp"]/message_doc[@name="ExpAverageConditional(Gamma, Gaussian, Gaussian)"]/*'/>
        public static Gamma ExpAverageConditional([NoInit] Gamma exp, [Proper] Gaussian d, [NoInit] Gaussian to_d)
        {
            if (d.IsPointMass)
                return Gamma.PointMass(Math.Exp(d.Point));
            if (d.IsUniform())
                return Gamma.FromShapeAndRate(0, 0);
            if (exp.IsPointMass)
            {
                // Z = int_y delta(x - exp(y)) N(y; my, vy) dy 
                //   = int_u delta(x - u) N(log(u); my, vy)/u du
                //   = N(log(x); my, vy)/x
                // logZ = -log(x) -0.5/vy*(log(x)-my)^2
                // dlogZ/dx = -1/x -1/vy*(log(x)-my)/x
                // d2logZ/dx2 = -dlogZ/dx/x -1/vy/x^2
                // log Ga(x;a,b) = (a-1)*log(x) - bx
                // dlogGa/dx = (a-1)/x - b
                // d2logGa/dx2 = -(a-1)/x^2
                // match derivatives and solve for (a,b)
                // a = 1 - d2logZ/dx2*x^2 = 1 + dlogZ/dx*x + 1/vy = -1/vy*(log(x)-my) + 1/vy
                // b = -d2logZ/dx2*x - dlogZ/dx = 1/vy/x
                double shape = d.MeanTimesPrecision + (1 - Math.Log(exp.Point)) * d.Precision;
                double rate = d.Precision / exp.Point;
                return Gamma.FromShapeAndRate(shape, rate);
            }
            if (exp.IsUniform())
                return ExpAverageLogarithm(d);

            if (to_d.IsUniform() && exp.Shape > 1)
            {
                to_d = new Gaussian(MMath.Digamma(exp.Shape - 1) - Math.Log(exp.Rate), MMath.Trigamma(exp.Shape - 1));
            }

            Gaussian dMarginal = d * to_d;
            dMarginal.GetMeanAndVariance(out double mD, out double vD);

            // Quadrature cannot get more than 6 digits of precision.
            // So in cases where extra precision is required, use another approach.
            if (vD < 1e-6)
            {
                //Trace.WriteLine($"vD < 1e-6");
                return ExpAverageLogarithm(d);
            }

            double Z = 0;
            double sumY = 0;
            double logsumExpY = double.NegativeInfinity;
            bool useHermite = true;
            //if (vD < 10)
            if (useHermite)
            {
                // Use Gauss-Hermite quadrature
                double[] nodes = new double[QuadratureNodeCount];
                double[] weights = new double[QuadratureNodeCount];

                Quadrature.GaussianNodesAndWeights(0, vD, nodes, weights);
                for (int i = 0; i < weights.Length; i++)
                {
                    weights[i] = Math.Log(weights[i]);
                }
                if (!to_d.IsUniform())
                {
                    // modify the weights to include q(y_k)/N(y_k;m,v)
                    for (int i = 0; i < weights.Length; i++)
                    {
                        weights[i] += d.GetLogProb(nodes[i] + mD) - Gaussian.GetLogProb(nodes[i], 0, vD);
                    }
                }

                double maxLogF = Double.NegativeInfinity;
                // f(x,y) = Ga(exp(y); shape, rate) = exp(y*(shape-1) -rate*exp(y))
                // Z E[x] = int_y int_x x Ga(x;a,b) delta(x - exp(y)) N(y;my,vy) dx dy
                //        = int_y exp(y) Ga(exp(y);a,b) N(y;my,vy) dy
                // Z E[log(x)] = int_y y Ga(exp(y);a,b) N(y;my,vy) dy
                double shapeMinus1 = exp.Shape - 1;
                for (int i = 0; i < weights.Length; i++)
                {
                    double y = nodes[i] + mD;
                    double logf = weights[i];
                    if (shapeMinus1 != 0) logf += shapeMinus1 * y; // avoid 0*inf
                    if (exp.Rate != 0) logf -= exp.Rate * Math.Exp(y); // avoid 0*inf
                    if (logf > maxLogF)
                    {
                        maxLogF = logf;
                    }
                    weights[i] = logf;
                }
                for (int i = 0; i < weights.Length; i++)
                {
                    double y = nodes[i];
                    double logf = weights[i] - maxLogF;
                    double f = Math.Exp(logf);
                    double f_y = f * y;
                    Z += f;
                    sumY += f_y;
                    logsumExpY = MMath.LogSumExp(logsumExpY, logf + y);
                }
            }
            else
            {
                double p(double y) { return d.GetLogProb(y) + (exp.Shape - 1) * y - exp.Rate * Math.Exp(y); }
                double sc = Math.Sqrt(vD);
                double offset = p(mD);
                double relTol = 1e-16;
                int nodeCount = 16 * 4;
                double scale = 1;
                Z = Quadrature.AdaptiveClenshawCurtis(z => Math.Exp(p(sc * z + mD) - offset), scale, nodeCount, relTol);
                sumY = Quadrature.AdaptiveClenshawCurtis(z => (sc * z) * Math.Exp(p(sc * z + mD) - offset), scale, nodeCount, relTol);
                double sumExpY = Quadrature.AdaptiveClenshawCurtis(z => Math.Exp(sc * z + p(sc * z + mD) - offset), scale, nodeCount, relTol);
                logsumExpY = Math.Log(sumExpY);
            }
            if (Z == 0)
                throw new InferRuntimeException("Z==0");
            double meanLogMinusMd = sumY / Z;
            double logMeanMinusMd = logsumExpY - Math.Log(Z);
            double logMeanMinusMeanLog = logMeanMinusMd - meanLogMinusMd;
            double mean = Math.Exp(logMeanMinusMd + mD);
            Gamma result = Gamma.FromLogMeanMinusMeanLog(mean, logMeanMinusMeanLog);
            //Trace.WriteLine($"result = {result} mean = {mean} logMeanMinusMeanLog = {logMeanMinusMeanLog}");
            result.SetToRatio(result, exp, ForceProper);
            if (Double.IsNaN(result.Shape) || Double.IsNaN(result.Rate))
                throw new InferRuntimeException($"result is NaN.  exp={exp}, d={d}, to_d={to_d}");
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp"]/message_doc[@name="ExpAverageConditionalInit(Gaussian)"]/*'/>
        [Skip]
        public static Gamma ExpAverageConditionalInit([IgnoreDependency] Gaussian d)
        {
            return Gamma.Uniform();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp"]/message_doc[@name="DAverageConditional(double)"]/*'/>
        public static Gaussian DAverageConditional(double exp)
        {
            return Gaussian.PointMass(Math.Log(exp));
        }

        //internal static Gaussian DAverageConditional_slow([SkipIfUniform] Gamma exp, [Proper] Gaussian d)
        //{
        //  Gaussian to_d = exp.Shape<=1 || exp.Rate==0 ? 
        //            Gaussian.Uniform() 
        //            : new Gaussian(MMath.Digamma(exp.Shape-1) - Math.Log(exp.Rate), MMath.Trigamma(exp.Shape));
        //  //var to_d = Gaussian.Uniform();
        //  for (int i = 0; i < QuadratureIterations; i++) {
        //    to_d = DAverageConditional(exp, d, to_d);
        //  }
        //  return to_d;
        //}
        // to_d does not need to be Fresh. it is only used for quadrature proposal.

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp"]/message_doc[@name="DAverageConditional(Gamma, Gaussian, Gaussian)"]/*'/>
        public static Gaussian DAverageConditional([SkipIfUniform] Gamma exp, [Proper] Gaussian d, Gaussian result)
        {
            if (exp.IsUniform() || d.IsUniform() || d.IsPointMass || exp.IsPointMass || exp.Rate <= 0)
                return ExpOp_Slow.DAverageConditional(exp, d);
            // We use moment matching to find the best Gaussian message.
            // The moments are computed via quadrature.
            // Z = int_y f(x,y) q(y) dy =approx sum_k w_k f(x,y_k) q(y_k)/N(y_k;m,v) 
            // f(x,y) = Ga(exp(y); shape, rate) = exp(y*(shape-1) -rate*exp(y))
            double[] nodes = new double[QuadratureNodeCount];
            double[] weights = new double[QuadratureNodeCount];
            d.GetMeanAndVariance(out double moD, out double voD);
            if (result.IsUniform() && exp.Shape > 1)
                result = new Gaussian(MMath.Digamma(exp.Shape - 1) - Math.Log(exp.Rate), MMath.Trigamma(exp.Shape - 1));
            Gaussian dMarginal = d * result;
            dMarginal.GetMeanAndVariance(out double mD, out double vD);
            if (vD == 0)
                return ExpOp_Slow.DAverageConditional(exp, d);
            // Do not add mD to the quadrature nodes because it will lose precision in the nodes when mD has large magnitude.
            Quadrature.GaussianNodesAndWeights(0, vD, nodes, weights);
            if (!result.IsUniform())
            {
                // modify the weights to include q(y_k)/N(y_k;m,v)
                for (int i = 0; i < weights.Length; i++)
                {
                    weights[i] *= Math.Exp(d.GetLogProb(nodes[i] + mD) - Gaussian.GetLogProb(nodes[i], 0, vD));
                }
            }
            double Z = 0;
            double sumy = 0;
            double sumy2 = 0;
            double maxLogF = Double.NegativeInfinity;
            double shapeMinus1 = exp.Shape - 1;
            for (int i = 0; i < weights.Length; i++)
            {
                double y = nodes[i] + mD;
                double logf = shapeMinus1 * y - exp.Rate * Math.Exp(y) + Math.Log(weights[i]);
                if (logf > maxLogF)
                {
                    maxLogF = logf;
                }
                weights[i] = logf;
            }
            for (int i = 0; i < weights.Length; i++)
            {
                double y = nodes[i];
                double f = Math.Exp(weights[i] - maxLogF);
                double f_y = f * y;
                double fyy = f_y * y;
                Z += f;
                sumy += f_y;
                sumy2 += fyy;
            }
            if (Z == 0)
                return Gaussian.Uniform();
            double meanMinusMode = sumy / Z;
            double mean = meanMinusMode + mD;
            // E[(y-mode)^2] = E[(y-mean)^2] + (mean - mode)^2
            double var = sumy2 / Z - meanMinusMode * meanMinusMode;
            if (var <= 0.0)
            {
                // Quadrature returns zero variance when the quadrature points are too far apart.
                // In this case, we estimate the variance based on the distance between quadrature points.
                double quadratureGap = 0.1;
                var = 2 * vD * quadratureGap * quadratureGap;
            }
            result = new Gaussian(mean, var);
            result.SetToRatio(result, d, ForceProper);
            if (result.Precision < -1e10)
                throw new InferRuntimeException("result has negative precision");
            if (Double.IsPositiveInfinity(result.Precision))
                throw new InferRuntimeException("result is point mass");
            if (Double.IsNaN(result.Precision) || Double.IsNaN(result.MeanTimesPrecision))
                return ExpOp_Slow.DAverageConditional(exp, d);
            return result;
        }

        //-- VMP -------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp"]/message_doc[@name="AverageLogFactor()"]/*'/>
        [Skip]
        public static double AverageLogFactor( /*[Proper] Gamma exp, [Proper] Gaussian d*/)
        {
            //double mean, variance;
            //d.GetMeanAndVariance(out mean, out variance);
            //return (exp.Shape - 1) * mean - exp.Rate * Math.Exp(mean + variance / 2)
            //    - MMath.GammaLn(exp.Shape) + exp.Shape * Math.Log(exp.Rate);
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp"]/message_doc[@name="ExpAverageLogarithmInit()"]/*'/>
        [Skip]
        public static Gamma ExpAverageLogarithmInit()
        {
            return Gamma.Uniform();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp"]/message_doc[@name="ExpAverageLogarithm(Gaussian)"]/*'/>
        public static Gamma ExpAverageLogarithm([Proper] Gaussian d)
        {
            if (d.IsPointMass) return Gamma.PointMass(Math.Exp(d.Point));
            d.GetMeanAndVariance(out double mD, out double vD);
            return ExpDistribution(mD, vD);
        }

        private static Gamma ExpDistribution(double mD, double vD)
        { 
            // E[exp(d)] = exp(mD + vD/2)
            double logMeanMinusMeanLog = vD / 2;
            double logMean = mD + logMeanMinusMeanLog;
            double mean = Math.Exp(logMean);
            //return Gamma.FromMeanAndVariance(Math.Exp(lm), Math.Exp(2*lm)*(Math.Exp(vD)-1));
            return Gamma.FromLogMeanMinusMeanLog(mean, logMeanMinusMeanLog);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp"]/message_doc[@name="ExpAverageLogarithm(NonconjugateGaussian)"]/*'/>
        [Quality(QualityBand.Experimental)]
        public static Gamma ExpAverageLogarithm([Proper] NonconjugateGaussian d)
        {
            d.GetMeanAndVariance(out double mD, out double vD);
            return ExpDistribution(mD, vD);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp"]/message_doc[@name="ExpAverageLogarithm(Gaussian, GammaPower)"]/*'/>
        public static GammaPower ExpAverageLogarithm([Proper] Gaussian d, GammaPower result)
        {
            if (d.IsPointMass) return GammaPower.PointMass(Math.Exp(d.Point), result.Power);
            d.GetMeanAndVariance(out double mD, out double vD);
            // E[exp(d)] = exp(mD + vD/2)
            double logMeanMinusMeanLog = vD / 2;
            double logMean = mD + logMeanMinusMeanLog;
            double mean = Math.Exp(logMean);
            //return Gamma.FromMeanAndVariance(Math.Exp(lm), Math.Exp(2*lm)*(Math.Exp(vD)-1));
            return GammaPower.FromLogMeanMinusMeanLog(mean, logMeanMinusMeanLog, result.Power);
        }

        // Finds the maximum of -C exp(v/2) + .5 log(v)
        // Should converge in <=10 iterations
        private static double FindMinimumInV(double C)
        {
            double THRESHOLD = 1E-3;
            int MAX_ITS = 30;
            double v = -Math.Log(C) + 0.31;
            if (C > 0.2178)
            {
                double old_v = v;
                for (int i = 0; i < MAX_ITS; i++)
                {
                    v = Math.Exp(-v / 2.0) / C;
                    if (Math.Abs(old_v - v) < THRESHOLD)
                        return v;
                    old_v = v;
                }
            }
            else if (C < 0.1576)
            {
                double old_v = v;
                for (int i = 0; i < MAX_ITS; i++)
                {
                    v = -2.0 * Math.Log(v * C);
                    if (v < 0)
                        throw new InferRuntimeException("FindMinimumInV failed for C= " + C);
                    if (Math.Abs(old_v - v) < THRESHOLD)
                        return v;
                    old_v = v;
                }
            }
            return v;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp"]/message_doc[@name="DAverageLogarithm(Gamma, NonconjugateGaussian, NonconjugateGaussian)"]/*'/>
        [Quality(QualityBand.Experimental)]
        public static NonconjugateGaussian DAverageLogarithm([Proper] Gamma exp, [Proper, SkipIfUniform] NonconjugateGaussian d, NonconjugateGaussian result)
        {
            return DNonconjugateAverageLogarithm(exp, d.GetGaussian(true), result);
        }

        /// <summary>
        /// Nonconjugate VMP message to 'd'.
        /// </summary>
        /// <param name="exp">Incoming message from 'exp'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
        /// <param name="d"></param>
        /// <param name="result"></param>
        /// <returns>The outgoing nonconjugate VMP message to the 'd' argument.</returns>
        /// <remarks><para>
        /// The outgoing message is the exponential of the integral of the log-factor times incoming messages, over all arguments except 'd'.
        /// The formula is <c>int log(f(d,x)) q(x) dx</c> where <c>x = (exp)</c>.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="exp"/> is not a proper distribution</exception>
        [Quality(QualityBand.Experimental)]
        public static NonconjugateGaussian DNonconjugateAverageLogarithm([Proper] Gamma exp, [Proper, SkipIfUniform] Gaussian d, NonconjugateGaussian result)
        {
            var vf = -1.0;
            var a = exp.Shape;
            var v_opt = 1.0 / (a - 1.0);
            var b = exp.Rate;
            d.GetMeanAndVariance(out double m, out double v);
            if (a > 1)
            {
                var mf = Math.Log((a - 1) / b) - .5 * v_opt;
                if (mf != m)
                {
                    var grad_S_m = -b * Math.Exp(m + v / 2) + a - 1;
                    vf = (mf - m) / grad_S_m;
                    result.MeanTimesPrecision = mf / vf;
                    result.Precision = 1 / vf;
                }
            }
            //m = (mp + prior_mp)/(p + prior_p);
            if (vf < 0)
            {
                result.Precision = b * Math.Exp(m + v / 2);
                result.MeanTimesPrecision = (m - 1) * result.Precision + a - 1;
            }

            double bf = -1, afm1 = -1;
            if (a <= 1)
                v_opt = FindMinimumInV(b * Math.Exp(m));
            if (v_opt != v)
            {
                var grad_S_v = -.5 * b * Math.Exp(m + v / 2);
                bf = v * grad_S_v / (v_opt - v);
                afm1 = v_opt * bf;
            }
            if (afm1 < 0 || bf < 0)
            {
                afm1 = b * v * v * Math.Exp(m + v / 2) / 4;
                bf = b * (1 + v / 2) * Math.Exp(m + v / 2) / 2;
            }
            result.Shape = afm1 + 1;
            result.Rate = bf;
            if (!result.IsProper())
                throw new InferRuntimeException("improper message calculated by ExpOp.DNonconjugateAverageLogarithm");
            return result;
        }

        public static bool UseRandomDamping = true;

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp"]/message_doc[@name="DAverageLogarithm(Gamma, Gaussian, Gaussian)"]/*'/>
        public static Gaussian DAverageLogarithm([Proper] Gamma exp, [Proper] Gaussian d, Gaussian to_d)
        {
            if (exp.IsPointMass)
                return DAverageLogarithm(exp.Point);

            d.GetMeanAndVariance(out double m, out double v);
            /* --------- This update comes from T. Minka's equations for non-conjugate VMP. -------- */
            Gaussian result = new Gaussian();
            result.Precision = exp.Rate * Math.Exp(m + v / 2);
            result.MeanTimesPrecision = (m - 1) * result.Precision + (exp.Shape - 1);
            if (UseRandomDamping)
            {
                double step = Rand.Double() * 0.5; // random damping helps convergence, especially with parallel updates
                result.Precision = step * result.Precision + (1 - step) * to_d.Precision;
                result.MeanTimesPrecision = step * result.MeanTimesPrecision + (1 - step) * to_d.MeanTimesPrecision;
            }
            if (result.IsPointMass)
                throw new Exception();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp"]/message_doc[@name="DAverageLogarithm(double)"]/*'/>
        public static Gaussian DAverageLogarithm(double exp)
        {
            return DAverageConditional(exp);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_Slow"]/doc/*'/>
    [FactorMethod(typeof(Math), "Exp", typeof(double))]
    [Quality(QualityBand.Experimental)]
    public static class ExpOp_Slow
    {
        public static int QuadratureNodeCount = 1000;

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_Slow"]/message_doc[@name="DAverageConditional(GammaPower, Gaussian)"]/*'/>
        public static Gaussian DAverageConditional([SkipIfUniform] GammaPower exp, [Proper] Gaussian d)
        {
            // as a function of d, the factor is Ga(exp(d); shape, rate, power) = exp(d*(shape/power -1) -rate*exp(d/power))
            // = Ga(exp(d/power); shape - power + 1, rate)
            double scale = 1 / exp.Power;
            Gaussian forward = GaussianProductOp.ProductAverageConditional(scale, d);
            Gaussian message = DAverageConditional(Gamma.FromNatural(exp.Shape - exp.Power, exp.Rate), forward);
            Gaussian backward = GaussianProductOp.BAverageConditional(message, scale);
            return backward;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_Slow"]/message_doc[@name="DAverageConditional(Gamma, double)"]/*'/>
        public static Gaussian DAverageConditional([SkipIfUniform] Gamma exp, double d)
        {
            double expx = Math.Exp(d);
            double ddlogf = -exp.Rate * expx;
            double dlogf = exp.Shape - 1 + ddlogf;
            // lim_{d -> -inf} d*exp(d) = 0
            if (d < double.MinValue) return Gaussian.FromNatural(dlogf, 0);
            // Gaussian.FromNatural(dlogf - ddlogf * d, -ddlogf)
            return Gaussian.FromDerivatives(d, dlogf, ddlogf, true);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_Slow"]/message_doc[@name="DAverageConditional(Gamma, Gaussian)"]/*'/>
        public static Gaussian DAverageConditional([SkipIfUniform] Gamma exp, [Proper] Gaussian d)
        {
            // as a function of d, the factor is Ga(exp(d); shape, rate) = exp(d*(shape-1) -rate*exp(d))
            if (exp.IsUniform())
                return Gaussian.Uniform();
            if (exp.IsPointMass)
                return ExpOp.DAverageConditional(exp.Point);
            if (exp.Rate == 0)
                return Gaussian.FromNatural(exp.Shape - 1, 0);
            if (d.IsPointMass)
            {
                return DAverageConditional(exp, d.Point);
            }
            if (exp.Rate < 0)
                throw new ImproperMessageException(exp);
            if (d.IsUniform())
            {
                if (exp.Shape <= 1)
                    throw new ArgumentException("The posterior has infinite variance due to input of Exp distributed as " + d + " and output of Exp distributed as " + exp +
                                                " (shape <= 1)");
                // posterior for d is a shifted log-Gamma distribution:
                // exp((a-1)*d - b*exp(d)) =propto exp(a*(d+log(b)) - exp(d+log(b)))
                // we find the Gaussian with same moments.
                // u = d+log(b)
                // E[u] = digamma(a-1)
                // E[d] = E[u]-log(b) = digamma(a-1)-log(b)
                // var(d) = var(u) = trigamma(a-1)
                double lnRate = Math.Log(exp.Rate);
                return new Gaussian(MMath.Digamma(exp.Shape - 1) - lnRate, MMath.Trigamma(exp.Shape - 1));
            }
            GetIntegrationBounds(exp, d, out double dmode, out double dminMinusMode, out double dmaxMinusMode);
            if (dminMinusMode == dmaxMinusMode) return DAverageConditional(exp, d.Point);
            double expmode = Math.Exp(dmode);
            int n = QuadratureNodeCount;
            double inc = (dmaxMinusMode - dminMinusMode) / (n - 1);
            MeanVarianceAccumulator mva = new MeanVarianceAccumulator();
            for (int i = 0; i < n; i++)
            {
                double xMinusMode = dminMinusMode + i * inc;
                double diff = LogFactorMinusMode(xMinusMode, exp, d, dmode, expmode);
                double p = Math.Exp(diff);
                mva.Add(xMinusMode, p);
                if (double.IsNaN(mva.Variance))
                    throw new Exception();
            }
            double dMean = mva.Mean + dmode;
            double dVariance = mva.Variance;
            Gaussian result = Gaussian.FromMeanAndVariance(dMean, dVariance);
            result.SetToRatio(result, d, true);
            return result;
        }

        public static double FindMaximum(Gamma exp, Gaussian d)
        {
            // find the mode
            // the mode satisfies: x = m + v*(a-1-b*exp(x))
            // with deriv = -v*b*exp(x)
            // or x = log((a-1 - (x-m)/v)/b)
            // with deriv = -1/((a-1)v - (x-m))
            d.GetMeanAndVariance(out double mx, out double vx);
            double aMinus1 = exp.Shape - 1;
            double b = exp.Rate;
            double x = mx;
            for (int iter = 0; iter < 1000; iter++)
            {
                double oldx = x;
                double deriv1 = vx * b * Math.Exp(x);
                double deriv2 = 1 / (aMinus1 * vx - (x - mx));
                if (deriv2 < 0 || deriv1 < deriv2)
                {
                    x = mx + vx * (aMinus1 - b * Math.Exp(x));
                }
                else
                {
                    x = Math.Log((aMinus1 - (x - mx) / vx) / b);
                }
                if (Math.Abs(x - oldx) < 1e-8)
                    break;
                // damping
                x = (x + oldx) / 2;
            }
            return x;
        }

        private static double LogFactorMinusMode(double xMinusMode, Gamma exp, Gaussian d, double mode, double expMode)
        {
            if (double.IsInfinity(xMinusMode))
                return double.NegativeInfinity;
            // Also see Gamma.GetLogProb
            double f = xMinusMode * (exp.Shape - 1 + (d.MeanTimesPrecision - mode * d.Precision) - 0.5 * xMinusMode * d.Precision) - exp.Rate * expMode * MMath.ExpMinus1(xMinusMode);
            if (double.IsNaN(f)) throw new Exception("f is NaN");
            return f;
        }

        public static void GetIntegrationBounds(Gamma exp, Gaussian d, out double dmode, out double dminMinusMode, out double dmaxMinusMode)
        {
            double mode = FindMaximum(exp, d);
            double expMode = Math.Exp(mode);
            // We want to find where LogFactorMinusMode == logUlp1, so we find zeroes of the difference.
            double logUlp1 = Math.Log(MMath.Ulp1);
            double func(double xMinusMode)
            {
                return LogFactorMinusMode(xMinusMode, exp, d, mode, expMode) - logUlp1;
            }
            double deriv(double xMinusMode)
            {
                return -exp.Rate * expMode * Math.Exp(xMinusMode) - xMinusMode * d.Precision + (d.MeanTimesPrecision - mode * d.Precision) + (exp.Shape - 1);
            }
            List<double> zeroes = GaussianOp_Slow.FindZeroes(func, deriv, new double[] { 0 }, new double[0]);
            dminMinusMode = MMath.Min(zeroes);
            dmaxMinusMode = MMath.Max(zeroes);
            dmode = mode;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_LaplaceProp"]/doc/*'/>
    [FactorMethod(typeof(Math), "Exp", typeof(double))]
    [Quality(QualityBand.Experimental)]
    public static class ExpOp_LaplaceProp
    {
        public static bool ForceProper;

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_LaplaceProp"]/message_doc[@name="LogAverageFactor(Gamma, Gaussian, Gaussian)"]/*'/>
        public static double LogAverageFactor([SkipIfUniform] Gamma exp, [Proper] Gaussian d, Gaussian to_d)
        {
            Gaussian dPost = d * to_d;
            double x = dPost.GetMean();
            double v = dPost.GetVariance();
            double expx = Math.Exp(x);
            double a = exp.Shape;
            double b = exp.Rate;
            return exp.GetLogProb(expx) + d.GetLogProb(x) + MMath.LnSqrt2PI + 0.5 * Math.Log(v);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_LaplaceProp"]/message_doc[@name="LogEvidenceRatio(Gamma, Gaussian, Gaussian, Gamma)"]/*'/>
        public static double LogEvidenceRatio([SkipIfUniform] Gamma exp, [Proper] Gaussian d, Gaussian to_d, Gamma to_exp)
        {
            return LogAverageFactor(exp, d, to_d) - to_exp.GetLogAverageOf(exp);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_LaplaceProp"]/message_doc[@name="DAverageConditional(Gamma, Gaussian, Gaussian)"]/*'/>
        public static Gaussian DAverageConditional([SkipIfUniform] Gamma exp, [Proper] Gaussian d, Gaussian to_d)
        {
            if (exp.IsPointMass)
                return ExpOp.DAverageConditional(exp.Point);
            Gaussian dPost = d * to_d;
            double dhat = dPost.GetMean();
            double ehat = Math.Exp(dhat);
            double a = exp.Shape;
            double b = exp.Rate;
            double dlogf = (a - 1) - b * ehat;
            double ddlogf = -b * ehat;
            double r = -ddlogf;
            if (ForceProper && r < 0)
                r = 0;
            return Gaussian.FromNatural(r * dhat + dlogf, r);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_LaplaceProp"]/message_doc[@name="ExpAverageConditional(Gamma, Gaussian, Gaussian)"]/*'/>
        public static Gamma ExpAverageConditional(Gamma exp, Gaussian d, Gaussian to_d)
        {
            if (d.IsPointMass)
                return Gamma.PointMass(Math.Exp(d.Point));
            if (exp.IsPointMass)
                return Gamma.Uniform();
            Gaussian dPost = d * to_d;
            double dhat = dPost.GetMean();
            double ehat = Math.Exp(dhat);
            double a = exp.Shape;
            double b = exp.Rate;
            double dlogf_diff = -ehat;
            double ddlogf_diff = 0;
            double ddlogfx = -ehat;
            double dlogz_diff = dlogf_diff;
            double dx = ddlogfx / (d.Precision - ddlogf_diff + a / (b * b));
            double ddlogz_diff = ddlogf_diff + ddlogfx * dx;
            double m = -dlogz_diff;
            double v = ddlogz_diff;
            Gamma result = Gamma.FromMeanAndVariance(m, v);
            result.SetToRatio(result, exp, ExpOp.ForceProper);
            return result;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_Laplace"]/doc/*'/>
    [FactorMethod(typeof(Math), "Exp", typeof(double))]
    [Buffers("x")]
    [Quality(QualityBand.Experimental)]
    public static class ExpOp_Laplace
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_Laplace"]/message_doc[@name="XInit(Gaussian)"]/*'/>
        public static double XInit([SkipIfUniform] Gaussian d)
        {
            return d.GetMean();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_Laplace"]/message_doc[@name="X2(Gamma, Gaussian, double)"]/*'/>
        public static double X2([SkipIfUniform] Gamma exp, [Proper] Gaussian d, double x)
        {
            // perform one Newton update of X
            double expx = Math.Exp(x);
            double a = exp.Shape;
            double b = exp.Rate;
            double r = b * expx;
            double dlogf = (a - 1) - r;
            //if (r < 0) r = 0;
            double t = r * x + dlogf;
            return (t + d.MeanTimesPrecision) / (r + d.Precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_Laplace"]/message_doc[@name="X(Gamma, Gaussian)"]/*'/>
        public static double X([SkipIfUniform] Gamma exp, [Proper] Gaussian d)
        {
            double x = 0;
            for (int iter = 0; iter < 1000; iter++)
            {
                double oldx = x;
                x = X2(exp, d, x);
                if (MMath.AbsDiff(x, oldx, 1e-10) < 1e-10) break;
            }
            return x;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_Laplace"]/message_doc[@name="LogAverageFactor(Gamma, Gaussian, double)"]/*'/>
        public static double LogAverageFactor([SkipIfUniform] Gamma exp, [Proper] Gaussian d, double x)
        {
            double expx = Math.Exp(x);
            double a = exp.Shape;
            double b = exp.Rate;
            double v = 1 / (d.Precision + b * expx);
            return exp.GetLogProb(expx) + d.GetLogProb(x) + MMath.LnSqrt2PI + 0.5 * Math.Log(v);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_Laplace"]/message_doc[@name="LogEvidenceRatio(Gamma, Gaussian, double, Gamma)"]/*'/>
        public static double LogEvidenceRatio([SkipIfUniform] Gamma exp, [Proper] Gaussian d, double x, Gamma to_exp)
        {
            return LogAverageFactor(exp, d, x) - to_exp.GetLogAverageOf(exp);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_Laplace"]/message_doc[@name="DAverageConditional(Gamma, Gaussian, double)"]/*'/>
        public static Gaussian DAverageConditional([SkipIfUniform] Gamma exp, [Proper] Gaussian d, double x)
        {
            if (exp.IsPointMass)
                return ExpOp.DAverageConditional(exp.Point);
            double expx = Math.Exp(x);
            double a = exp.Shape;
            double b = exp.Rate;
            double ddlogf = -b * expx;
            double dddlogf = ddlogf;
            double d4logf = ddlogf;
            double dlogf = (a - 1) + ddlogf;
            double v = 1 / (d.Precision - ddlogf);
            // this is Laplace's estimate of the posterior mean and variance
            double mpost = x + 0.5 * dddlogf * v * v;
            double vpost = v + 0.5 * d4logf * v * v * v + dddlogf * dddlogf * v * v * v * v;
            //double vpost = v + 0.25 * dddlogf * dddlogf * v * v * v * v;
            Gaussian result = Gaussian.FromMeanAndVariance(mpost, vpost);
            result.SetToRatio(result, d, ExpOp.ForceProper);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_Laplace"]/message_doc[@name="ExpAverageConditional(Gamma, Gaussian, double)"]/*'/>
        public static Gamma ExpAverageConditional(Gamma exp, Gaussian d, double x)
        {
            if (d.IsPointMass)
                return Gamma.PointMass(Math.Exp(d.Point));
            if (exp.IsPointMass)
                return Gamma.Uniform();
            double a = exp.Shape;
            double b = exp.Rate;
            double expx = Math.Exp(x);
            //double apost = a - (1 + d.Precision*(x-1) - d.MeanTimesPrecision);
            //double bpost = b + d.Precision/expx;
            //double mpost = expx - d.Precision*(MMath.Digamma(apost) - Math.Log(apost))/bpost;
            double v = 1 / (d.Precision + b * expx);
            double meanLogMinusX = -0.5 * v * v * b * expx;
            double meanLog = x + meanLogMinusX;
            double logMeanMinusX = Math.Log(1 + 0.5 * v * v * d.Precision);
            double logMeanMinusMeanLog = logMeanMinusX - meanLogMinusX;
            double logMean = x + logMeanMinusX;
            double mean = Math.Exp(logMean);
            Gamma result = Gamma.FromLogMeanMinusMeanLog(mean, logMeanMinusMeanLog);
            result.SetToRatio(result, exp, true);
            return result;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_BFGS"]/doc/*'/>
    [FactorMethod(typeof(Math), "Exp", typeof(double))]
    [Quality(QualityBand.Preview)]
    public class ExpOp_BFGS
    {
        private static double GradientAndValueAtPoint(double mu, double s2, Vector x, double a, double b, Vector grad)
        {
            double m, v;
            m = x[0];
            v = Math.Exp(x[1]);
            double kl_value = -.5 * (1.0 + x[1]) + .5 * (v + (m - mu) * (m - mu)) / s2 + .5 * Math.Log(s2)
                              - ((a - 1) * m - b * Math.Exp(m + v / 2.0)); // + const
            if (grad != null)
            {
                grad[0] = (m - mu) / s2 - ((a - 1) - b * Math.Exp(m + v / 2.0));
                grad[1] = -.5 + .5 * v / s2 + b * .5 * v * Math.Exp(m + v / 2.0);
            }
            return kl_value;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExpOp_BFGS"]/message_doc[@name="DAverageLogarithm(Gamma, Gaussian, Gaussian)"]/*'/>
        public static Gaussian DAverageLogarithm([Proper] Gamma exp, [Proper, Stochastic] Gaussian d, Gaussian to_d)
        {
            if (exp.IsPointMass)
                return ExpOp.DAverageLogarithm(exp.Point);

            d.GetMeanAndVariance(out double m, out double v);
            var prior = d / to_d;
            prior.GetMeanAndVariance(out double mu, out double s2);
            var z = Vector.Zero(2);
            z[0] = m;
            z[1] = Math.Log(v);
            double startingValue = GradientAndValueAtPoint(mu, s2, z, exp.Shape, exp.Rate, null);
            var s = new BFGS();
            int evalCounter = 0;
            s.MaximumStep = 1e3;
            s.MaximumIterations = 100;
            s.Epsilon = 1e-5;
            s.convergenceCriteria = BFGS.ConvergenceCriteria.Objective;
            z = s.Run(z, 1.0, delegate (Vector y, ref Vector grad)
                {
                    evalCounter++;
                    return GradientAndValueAtPoint(mu, s2, y, exp.Shape, exp.Rate, grad);
                });
            m = z[0];
            v = Math.Exp(z[1]);
            to_d.SetMeanAndVariance(m, v);
            to_d.SetToRatio(to_d, prior);
            double endValue = GradientAndValueAtPoint(mu, s2, z, exp.Shape, exp.Rate, null);
            //Console.WriteLine("Went from {0} to {1} in {2} steps, {3} evals", startingValue, endValue, s.IterationsPerformed, evalCounter);
            if (startingValue < endValue)
                Console.WriteLine("Warning: BFGS resulted in an increased objective function");
            return to_d;

            /* ---------------- NEWTON ITERATION VERSION 1 ------------------- 
            double meanTimesPrec, prec;
            d.GetNatural(out meanTimesPrec, out prec);
            Matrix K = new Matrix(2, 2);
            K[0, 0]=1/prec; // d2K by dmu^2
            K[1, 0]=K[0, 1]=-meanTimesPrec/(prec*prec);
            K[1, 1]=meanTimesPrec*meanTimesPrec/Math.Pow(prec, 3)+1/(2*prec*prec);
            double[,,] Kprime = new double[2, 2, 2];
            Kprime[0, 0, 0]=0;
            Kprime[0, 0, 1]=Kprime[0, 1, 0]=Kprime[1, 0, 0]=-1/(prec*prec);
            Kprime[0, 1, 1]=Kprime[1, 1, 0]=Kprime[1, 0, 1]=2*meanTimesPrec/Math.Pow(prec, 3);
            Kprime[1, 1, 1]=-3*meanTimesPrec*meanTimesPrec/Math.Pow(prec, 4)-1/Math.Pow(prec, 3);
            Vector gradS = new Vector(2);
            gradS[0]=(exp.Shape-1)/prec-exp.Rate/prec*Math.Exp((meanTimesPrec+.5)/prec);
            gradS[1]=-(exp.Shape-1)*meanTimesPrec/(prec*prec)+exp.Rate*(meanTimesPrec+.5)/(prec*prec)*Math.Exp((meanTimesPrec+.5)/prec);
            Matrix grad2S = new Matrix(2, 2);
            grad2S[0, 0]=-exp.Rate/(prec*prec)*Math.Exp((meanTimesPrec+.5)/prec);
            grad2S[0, 1]=grad2S[1, 0]=-(exp.Shape-1)/(prec*prec)+exp.Rate*(1/(prec*prec)+(meanTimesPrec+.5)/Math.Pow(prec, 3))*Math.Exp((meanTimesPrec+.5)/prec);
            grad2S[1, 1]=2*(exp.Shape-1)*meanTimesPrec/Math.Pow(prec, 3)-exp.Rate*(meanTimesPrec+.5)/(prec*prec)*(2/prec+(meanTimesPrec+.5)/(prec*prec))*Math.Exp((meanTimesPrec+.5)/prec);
            Vector phi = new Vector(new double[] { result.MeanTimesPrecision, result.Precision });
            Vector gradKL = K*phi-gradS;
            Matrix hessianKL = K - grad2S;
            for (int i=0; i<2; i++)
                            for (int j=0; j<2; j++)
                                            for (int k=0; k<2; k++)
                                                            hessianKL[i, j]+=Kprime[i, j, k]*phi[k];
            double step = 1;
            Vector direction = GammaFromShapeAndRate.twoByTwoInverse(hessianKL)*gradKL;
            Vector newPhi = phi - step * direction;
            result.SetNatural(newPhi[0], newPhi[1]);
            return result;

            ---------------- NEWTON ITERATION VERSION 2 ------------------- 
            double mean, variance;
            d.GetMeanAndVariance(out mean, out variance); 
            Gaussian prior = d / result; 
            double mean1, variance1;
            prior.GetMeanAndVariance(out mean1, out variance1); 
            Vector gradKL = new Vector(2);
            gradKL[0]=-(exp.Shape-1)+exp.Rate*Math.Exp(mean+variance/2)+mean/variance1-mean1/variance1;
            gradKL[1]=-1/(2*variance)+exp.Rate*Math.Exp(mean+variance/2)+1/(2*variance1);
            Matrix hessianKL = new Matrix(2, 2);
            hessianKL[0, 0]=exp.Rate*Math.Exp(mean+variance/2)+1/variance1;
            hessianKL[0, 1]=hessianKL[1, 0]=.5*exp.Rate*Math.Exp(mean+variance/2);
            hessianKL[1, 1]=1/(2*variance*variance)+exp.Rate*Math.Exp(mean+variance/2)/4;
            result.GetMeanAndVariance(out mean, out variance);
            if (double.IsInfinity(variance))
                            variance=1000;
            Vector theta = new Vector(new double[] { mean, variance });
            theta -= GammaFromShapeAndRate.twoByTwoInverse(hessianKL)*gradKL;
            result.SetMeanAndVariance(theta[0], theta[1]);
            return result; 
            ----------------------------------------------------------------- */
        }
    }
}
