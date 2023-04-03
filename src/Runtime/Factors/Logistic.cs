// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogisticOp"]/doc/*'/>
    [FactorMethod(typeof(MMath), "Logistic", typeof(double))]
    [Quality(QualityBand.Stable)]
    [Buffers("falseMsg")]
    public class LogisticOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogisticOp"]/message_doc[@name="LogAverageFactor(double, double)"]/*'/>
        public static double LogAverageFactor(double logistic, double x)
        {
            return (logistic == MMath.Logistic(x)) ? 0.0 : Double.NegativeInfinity;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogisticOp"]/message_doc[@name="LogEvidenceRatio(double, double)"]/*'/>
        public static double LogEvidenceRatio(double logistic, double x)
        {
            return LogAverageFactor(logistic, x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogisticOp"]/message_doc[@name="AverageLogFactor(double, double)"]/*'/>
        public static double AverageLogFactor(double logistic, double x)
        {
            return LogAverageFactor(logistic, x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogisticOp"]/message_doc[@name="LogAverageFactor(Beta, double)"]/*'/>
        public static double LogAverageFactor(Beta logistic, double x)
        {
            return logistic.GetLogProb(MMath.Logistic(x));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogisticOp"]/message_doc[@name="LogAverageFactor(double, Gaussian)"]/*'/>
        public static double LogAverageFactor(double logistic, Gaussian x)
        {
            if (logistic >= 1.0 || logistic <= 0.0)
                return x.GetLogProb(MMath.Logit(logistic));
            // p(y,x) = delta(y - 1/(1+exp(-x))) N(x;mx,vx)
            // x = log(y/(1-y))
            // dx = 1/(y*(1-y))
            return x.GetLogProb(MMath.Logit(logistic)) / (logistic * (1 - logistic));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogisticOp"]/message_doc[@name="LogEvidenceRatio(double, Gaussian)"]/*'/>
        public static double LogEvidenceRatio(double logistic, Gaussian x)
        {
            return LogAverageFactor(logistic, x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogisticOp"]/message_doc[@name="LogAverageFactor(Beta, Gaussian, Gaussian)"]/*'/>
        public static double LogAverageFactor(Beta logistic, Gaussian x, Gaussian falseMsg)
        {
            // return log(int_y int_x delta(y - Logistic(x)) Beta(y) Gaussian(x) dx dy)
            double m, v;
            x.GetMeanAndVariance(out m, out v);
            if (logistic.TrueCount == 2 && logistic.FalseCount == 1)
            {
                // shortcut for common case
                return Math.Log(2 * MMath.LogisticGaussian(m, v));
            }
            else if (logistic.TrueCount == 1 && logistic.FalseCount == 2)
            {
                return Math.Log(2 * MMath.LogisticGaussian(-m, v));
            }
            else
            {
                // logistic(sigma(x)) N(x;m,v)
                // = sigma(x)^(a-1) sigma(-x)^(b-1) N(x;m,v) gamma(a+b)/gamma(a)/gamma(b)
                // = e^((a-1)x) sigma(-x)^(a+b-2) N(x;m,v)
                // = sigma(-x)^(a+b-2) N(x;m+(a-1)v,v) exp((a-1)m + (a-1)^2 v/2)
                // int_x logistic(sigma(x)) N(x;m,v) dx 
                // =approx (int_x sigma(-x)/falseMsg(x) falseMsg(x)^(a+b-2) N(x;m+(a-1)v,v))^(a+b-2) 
                //       * (int_x falseMsg(x)^(a+b-2) N(x;m+(a-1)v,v))^(1 - (a+b-2))
                //       *  exp((a-1)m + (a-1)^2 v/2) gamma(a+b)/gamma(a)/gamma(b)
                // This formula comes from (66) in Minka (2005)
                // Alternatively,
                // =approx (int_x falseMsg(x)/sigma(-x) falseMsg(x)^(a+b-2) N(x;m+(a-1)v,v))^(-(a+b-2))
                //       * (int_x falseMsg(x)^(a+b-2) N(x;m+(a-1)v,v))^(1 + (a+b-2))
                //       *  exp((a-1)m + (a-1)^2 v/2) gamma(a+b)/gamma(a)/gamma(b)
                double tc1 = logistic.TrueCount - 1;
                double fc1 = logistic.FalseCount - 1;
                Gaussian prior = new Gaussian(m + tc1 * v, v);
                if (tc1 + fc1 < 0)
                {
                    // numerator2 = int_x falseMsg(x)^(a+b-1) N(x;m+(a-1)v,v) dx
                    double numerator2 = prior.GetLogAverageOfPower(falseMsg, tc1 + fc1 + 1);
                    Gaussian prior2 = prior * (falseMsg ^ (tc1 + fc1 + 1));
                    double mp, vp;
                    prior2.GetMeanAndVariance(out mp, out vp);
                    // numerator = int_x (1+exp(x)) falseMsg(x)^(a+b-1) N(x;m+(a-1)v,v) dx / int_x falseMsg(x)^(a+b-1) N(x;m+(a-1)v,v) dx
                    double numerator = Math.Log(1 + Math.Exp(mp + 0.5 * vp));
                    // denominator = int_x falseMsg(x)^(a+b-2) N(x;m+(a-1)v,v) dx
                    double denominator = prior.GetLogAverageOfPower(falseMsg, tc1 + fc1);
                    return -(tc1 + fc1) * (numerator + numerator2 - denominator) + denominator + (tc1 * m + tc1 * tc1 * v * 0.5) - logistic.GetLogNormalizer();
                }
                else
                {
                    // numerator2 = int_x falseMsg(x)^(a+b-3) N(x;m+(a-1)v,v) dx
                    double numerator2 = prior.GetLogAverageOfPower(falseMsg, tc1 + fc1 - 1);
                    Gaussian prior2 = prior * (falseMsg ^ (tc1 + fc1 - 1));
                    double mp, vp;
                    prior2.GetMeanAndVariance(out mp, out vp);
                    // numerator = int_x sigma(-x) falseMsg(x)^(a+b-3) N(x;m+(a-1)v,v) dx / int_x falseMsg(x)^(a+b-3) N(x;m+(a-1)v,v) dx
                    double numerator = Math.Log(MMath.LogisticGaussian(-mp, vp));
                    // denominator = int_x falseMsg(x)^(a+b-2) N(x;m+(a-1)v,v) dx
                    double denominator = prior.GetLogAverageOfPower(falseMsg, tc1 + fc1);
                    return (tc1 + fc1) * (numerator + numerator2 - denominator) + denominator + (tc1 * m + tc1 * tc1 * v * 0.5) - logistic.GetLogNormalizer();
                }
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogisticOp"]/message_doc[@name="LogEvidenceRatio(Beta, Gaussian, Gaussian, Beta)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Beta logistic, Gaussian x, Gaussian falseMsg, [Fresh] Beta to_logistic)
        {
            // always zero when using the stabilized message from LogisticAverageConditional
            return 0.0;
            //return LogAverageFactor(logistic, x, falseMsg) - to_logistic.GetLogAverageOf(logistic);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogisticOp"]/message_doc[@name="LogisticAverageConditionalInit()"]/*'/>
        [Skip]
        public static Beta LogisticAverageConditionalInit()
        {
            return Beta.Uniform();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogisticOp"]/message_doc[@name="LogisticAverageConditional(Beta, Gaussian, Gaussian, Gaussian)"]/*'/>
        public static Beta LogisticAverageConditional(Beta logistic, [Proper] Gaussian x, Gaussian falseMsg, Gaussian to_x)
        {
            if (x.IsPointMass)
                return Beta.PointMass(MMath.Logistic(x.Point));

            if (logistic.IsPointMass || x.IsUniform())
                return Beta.Uniform();

            Gaussian post = to_x * x;
            double m, v;
            post.GetMeanAndVariance(out m, out v);
            double mean = MMath.LogisticGaussian(m, v);
            bool useVariance = logistic.IsUniform();  // useVariance gives lower accuracy on tests, but is required for the uniform case
            if (useVariance)
            {
                // meanTF = E[p] - E[p^2]
                double meanTF = MMath.LogisticGaussianDerivative(m, v);
                double meanSquare = mean - meanTF;
                Beta result = Beta.FromMeanAndVariance(mean, meanSquare - mean * mean);
                result.SetToRatio(result, logistic, true);
                return result;
            }
            else
            {
                double logZ = LogAverageFactor(logistic, x, falseMsg) + logistic.GetLogNormalizer(); // log int_x logistic(sigma(x)) N(x;m,v) dx
                double tc1 = logistic.TrueCount - 1;
                double fc1 = logistic.FalseCount - 1;
                return BetaFromMeanAndIntegral(mean, logZ, tc1, fc1);
            }
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        /// <summary>
        /// Find a Beta distribution with given integral and mean times a Beta weight function.
        /// </summary>
        /// <param name="mean">The desired value of the mean</param>
        /// <param name="logZ">The desired value of the integral</param>
        /// <param name="a">trueCount-1 of the weight function</param>
        /// <param name="b">falseCount-1 of the weight function</param>
        /// <returns></returns>
        private static Beta BetaFromMeanAndIntegral(double mean, double logZ, double a, double b)
        {
            // The constraints are:
            // 1. int_p to_p(p) p^a (1-p)^b dp = exp(logZ)
            // 2. int_p to_p(p) p p^a (1-p)^b dp = mean*exp(logZ)
            // Let to_p(p) = Beta(p; af, bf)
            // The LHS of (1) is gamma(af+bf)/gamma(af+bf+a+b) gamma(af+a)/gamma(af) gamma(bf+b)/gamma(bf)
            // The LHS of (2) is gamma(af+bf)/gamma(af+bf+a+b+1) gamma(af+a+1)/gamma(af) gamma(bf+b)/gamma(bf)
            // The ratio of (2)/(1) is gamma(af+a+1)/gamma(af+a) gamma(af+bf+a+b)/gamma(af+bf+a+b+1) = (af+a)/(af+bf+a+b) = mean
            // Solving for bf gives bf = (af+a)/mean - (af+a+b).
            // To solve for af, we apply a generalized Newton algorithm to solve equation (1) with bf substituted.
            // af0 is the smallest value of af that ensures (af >= 0, bf >= 0).
            if (mean <= 0)
                throw new ArgumentException("mean <= 0");
            if (mean >= 1)
                throw new ArgumentException("mean >= 1");
            if (double.IsNaN(mean))
                throw new ArgumentException("mean is NaN");
            // If exp(logZ) exceeds the largest possible value of (1), then we return a point mass.
            // gammaln(x) =approx (x-0.5)*log(x) - x + 0.5*log(2pi)
            // (af+x)*log(af+x) =approx (af+x)*log(af) + x + 0.5*x*x/af
            // For large af, logZ = (af+bf-0.5)*log(af+bf) - (af+bf+a+b-0.5)*log(af+bf+a+b) + 
            //                      (af+a-0.5)*log(af+a) - (af-0.5)*log(af) +
            //                      (bf+b-0.5)*log(bf+b) - (bf-0.5)*log(bf)
            // =approx (af+bf-0.5)*log(af+bf) - ((af+bf+a+b-0.5)*log(af+bf) + (a+b) + 0.5*(a+b)*(a+b)/(af+bf) -0.5*(a+b)/(af+bf)) + 
            //   ((af+a-0.5)*log(af) + a + 0.5*a*a/af - 0.5*a/af) - (af-0.5)*log(af) +
            //   ((bf+b-0.5)*log(bf) + b + 0.5*b*b/bf - 0.5*b/bf) - (bf-0.5)*log(bf)
            // = -(a+b)*log(af+bf) - 0.5*(a+b)*(a+b-1)/(af+bf) + a*log(af) + 0.5*a*(a-1)/af + b*log(bf) + 0.5*b*(b-1)/bf
            // =approx (a+b)*log(m) + b*log((1-m)/m) + 0.5*(a+b)*(a+b-1)*m/af - 0.5*a*(a+1)/af - 0.5*b*(b+1)*m/(1-m)/af
            // =approx (a+b)*log(mean) + b*log((1-mean)/mean)
            double maxLogZ = (a + b) * Math.Log(mean) + b * Math.Log((1 - mean) / mean);
            // slope determines whether maxLogZ is the maximum or minimum possible value of logZ
            double slope = (a + b) * (a + b - 1) * mean - a * (a + 1) - b * (b + 1) * mean / (1 - mean);
            if ((slope <= 0 && logZ >= maxLogZ) || (slope > 0 && logZ <= maxLogZ))
            {
                // optimal af is infinite
                return Beta.PointMass(mean);
            }
            // bf = (af+bx)*(1-m)/m
            double bx = -(mean * (a + b) - a) / (1 - mean);
            // af0 is the lower bound for af
            // we need both af>0 and bf>0
            double af0 = Math.Max(0, -bx);
            double x = Math.Max(0, bx);
            double af = af0 + 1; // initial guess for af
            double invMean = 1 / mean;
            double bf = (af + a) * invMean - (af + a + b);
            int numIters = 20;
            for (int iter = 0; iter < numIters; iter++)
            {
                double old_af = af;
                double f = (MMath.GammaLn(af + bf) - MMath.GammaLn(af + bf + a + b)) + (MMath.GammaLn(af + a) - MMath.GammaLn(af)) +
                           (MMath.GammaLn(bf + b) - MMath.GammaLn(bf));
                double g = (MMath.Digamma(af + bf) - MMath.Digamma(af + bf + a + b)) * invMean + (MMath.Digamma(af + a) - MMath.Digamma(af)) +
                           (MMath.Digamma(bf + b) - MMath.Digamma(bf)) * (invMean - 1);
                // fit a fcn of the form: s*log((af-af0)/(af+x)) + c
                // whose deriv is s/(af-af0) - s/(af+x)
                double s = g / (1 / (af - af0) - 1 / (af + x));
                double c = f - s * Math.Log((af - af0) / (af + x));
                bool isIncreasing = (x > -af0);
                if ((!isIncreasing && c >= logZ) || (isIncreasing && c <= logZ))
                {
                    // the approximation doesn't fit; use Gauss-Newton instead
                    af += (logZ - f) / g;
                }
                else
                {
                    // now solve s*log((af-af0)/(af+x))+c = logz
                    // af-af0 = exp((logz-c)/s) (af+x)
                    af = af0 + (x + af0) / MMath.ExpMinus1((c - logZ) / s);
                    //if (af == af0)
                    //    throw new ArgumentException("logZ is out of range");
                }
                if (double.IsNaN(af))
                    throw new InferRuntimeException("af is nan");
                bf = (af + a) / mean - (af + a + b);
                if (Math.Abs(af - old_af) < 1e-8 || af == af0)
                    break;
                //if (iter == numIters-1)
                //    throw new Exception("not converging");
            }
            if (false)
            {
                // check that integrals are correct
                double f = (MMath.GammaLn(af + bf) - MMath.GammaLn(af + bf + a + b)) + (MMath.GammaLn(af + a) - MMath.GammaLn(af)) +
                           (MMath.GammaLn(bf + b) - MMath.GammaLn(bf));
                if (Math.Abs(f - logZ) > 1e-6)
                    throw new InferRuntimeException("wrong f");
                double f2 = (MMath.GammaLn(af + bf) - MMath.GammaLn(af + bf + a + b + 1)) + (MMath.GammaLn(af + a + 1) - MMath.GammaLn(af)) +
                            (MMath.GammaLn(bf + b) - MMath.GammaLn(bf));
                if (Math.Abs(f2 - (Math.Log(mean) + logZ)) > 1e-6)
                    throw new InferRuntimeException("wrong f2");
            }
            return new Beta(af, bf);
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogisticOp"]/message_doc[@name="XAverageConditional(double)"]/*'/>
        public static Gaussian XAverageConditional(double logistic)
        {
            return Gaussian.PointMass(MMath.Logit(logistic));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogisticOp"]/message_doc[@name="FalseMsgInit()"]/*'/>
        [Skip]
        public static Gaussian FalseMsgInit()
        {
            return new Gaussian();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogisticOp"]/message_doc[@name="FalseMsg(Beta, Gaussian, Gaussian)"]/*'/>
        public static Gaussian FalseMsg([SkipIfUniform] Beta logistic, [Proper] Gaussian x, Gaussian falseMsg)
        {
            if (x.IsUniform()) throw new ArgumentException("x is uniform", nameof(x));
            // falseMsg approximates sigma(-x)
            // logistic(sigma(x)) N(x;m,v)
            // = sigma(x)^(a-1) sigma(-x)^(b-1) N(x;m,v) 
            // = e^((a-1)x) sigma(-x)^(a+b-2) N(x;m,v)
            // = sigma(-x)^(a+b-2) N(x;m+(a-1)v,v) exp((a-1)m + (a-1)^2 v/2)
            // = sigma(-x) (prior)
            // where prior = sigma(-x)^(a+b-3) N(x;m+(a-1)v,v)
            double tc1 = logistic.TrueCount - 1;
            double fc1 = logistic.FalseCount - 1;
            double m, v;
            x.GetMeanAndVariance(out m, out v);
            if (tc1 + fc1 == 0)
            {
                falseMsg.SetToUniform();
                return falseMsg;
            }
            else if (tc1 + fc1 < 0)
            {
                // power EP update, using 1/sigma(-x) as the factor
                Gaussian prior = new Gaussian(m + tc1 * v, v) * (falseMsg ^ (tc1 + fc1 + 1));
                double mprior, vprior;
                prior.GetMeanAndVariance(out mprior, out vprior);
                // posterior moments can be computed exactly
                double w = MMath.Logistic(mprior + 0.5 * vprior);
                Gaussian post = new Gaussian(mprior + w * vprior, vprior * (1 + w * (1 - w) * vprior));
                return prior / post;
            }
            else
            {
                // power EP update
                Gaussian prior = new Gaussian(m + tc1 * v, v) * (falseMsg ^ (tc1 + fc1 - 1));
                Gaussian newMsg = BernoulliFromLogOddsOp.LogOddsAverageConditional(false, prior);
                //Console.WriteLine("prior = {0}, falseMsg = {1}, newMsg = {2}", prior, falseMsg, newMsg);
                if (string.Empty.Length==0)
                {
                    // adaptive damping scheme
                    Gaussian ratio = newMsg / falseMsg;
                    if ((ratio.MeanTimesPrecision < 0 && prior.MeanTimesPrecision > 0) ||
                        (ratio.MeanTimesPrecision > 0 && prior.MeanTimesPrecision < 0))
                    {
                        // if the update would change the sign of the mean, take a fractional step so that the new prior has exactly zero mean
                        // newMsg = falseMsg * (ratio^step)
                        // newPrior = prior * (ratio^step)^(tc1+fc1-1)
                        // 0 = prior.mp + ratio.mp*step*(tc1+fc1-1)
                        double step = -prior.MeanTimesPrecision / (ratio.MeanTimesPrecision * (tc1 + fc1 - 1));
                        if (step > 0 && step < 1)
                        {
                            newMsg = falseMsg * (ratio ^ step);
                            // check that newPrior has zero mean
                            //Gaussian newPrior = prior * ((ratio^step)^(tc1+fc1-1));
                            //Console.WriteLine(newPrior);
                        }
                    }
                }
                else
                {
                    for (int iter = 0; iter < 10; iter++)
                    {
                        newMsg = falseMsg * ((newMsg / falseMsg) ^ 0.5);
                        falseMsg = newMsg;
                        //Console.WriteLine("prior = {0}, falseMsg = {1}, newMsg = {2}", prior, falseMsg, newMsg);
                        prior = new Gaussian(m + tc1 * v, v) * (falseMsg ^ (tc1 + fc1 - 1));
                        newMsg = BernoulliFromLogOddsOp.LogOddsAverageConditional(false, prior);
                    }
                }
                return newMsg;
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogisticOp"]/message_doc[@name="XAverageConditional(Beta, Gaussian)"]/*'/>
        public static Gaussian XAverageConditional([SkipIfUniform] Beta logistic, Gaussian falseMsg)
        {
            if (logistic.IsPointMass)
                return XAverageConditional(logistic.Point);
            if (falseMsg.IsPointMass)
                throw new ArgumentException("falseMsg is a point mass");
            // sigma(x)^(a-1) sigma(-x)^(b-1)
            // = e^((a-1)x) falseMsg^(a+b-2)
            // e^((a-1)x) = Gaussian.FromNatural(a-1,0)
            double tc1 = logistic.TrueCount - 1;
            double fc1 = logistic.FalseCount - 1;
            return Gaussian.FromNatural((tc1 + fc1) * falseMsg.MeanTimesPrecision + tc1, (tc1 + fc1) * falseMsg.Precision);
        }

        //-- VMP -------------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogisticOp"]/message_doc[@name="AverageLogFactor(Gaussian, Beta, Beta)"]/*'/>
        //[Skip]
        public static double AverageLogFactor([Proper, SkipIfUniform] Gaussian x, Beta logistic, Beta to_logistic)
        {
            double m, v;
            x.GetMeanAndVariance(out m, out v);
            double l1pe = v == 0 ? MMath.Log1PlusExp(m) : MMath.Log1PlusExpGaussian(m, v);
            return (logistic.TrueCount - 1.0) * (m - l1pe) + (logistic.FalseCount - 1.0) * (-l1pe) - logistic.GetLogNormalizer() - to_logistic.GetAverageLog(logistic);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogisticOp"]/message_doc[@name="LogisticAverageLogarithm(Gaussian)"]/*'/>
        public static Beta LogisticAverageLogarithm([Proper] Gaussian x)
        {
            double m, v;
            x.GetMeanAndVariance(out m, out v);

#if true
            // for consistency with XAverageLogarithm
            double eLogOneMinusP = BernoulliFromLogOddsOp.AverageLogFactor(false, x);
#else
    // E[log (1-sigma(x))] = E[log sigma(-x)] = -E[log(1+exp(x))]
            double eLogOneMinusP = -MMath.Log1PlusExpGaussian(m, v);
#endif
            // E[log sigma(x)] = -E[log(1+exp(-x))] = -E[log(1+exp(x))-x] = -E[log(1+exp(x))] + E[x]
            double eLogP = eLogOneMinusP + m;
            return Beta.FromMeanLogs(eLogP, eLogOneMinusP);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogisticOp"]/message_doc[@name="XAverageLogarithm(Beta, Gaussian, Gaussian)"]/*'/>
        public static Gaussian XAverageLogarithm([SkipIfUniform] Beta logistic, [Proper, SkipIfUniform] Gaussian x, Gaussian to_X)
        {
            if (logistic.IsPointMass)
                return XAverageLogarithm(logistic.Point);
            // f(x) = sigma(x)^(a-1) sigma(-x)^(b-1)
            //      = sigma(x)^(a+b-2) exp(-x(b-1))
            // since sigma(-x) = sigma(x) exp(-x)

            double a = logistic.TrueCount;
            double b = logistic.FalseCount;
            double scale = a + b - 2;
            if (scale == 0.0)
                return Gaussian.Uniform();
            double shift = -(b - 1);
            Gaussian toLogOddsPrev = Gaussian.FromNatural((to_X.MeanTimesPrecision - shift) / scale, to_X.Precision / scale);
            Gaussian toLogOdds = BernoulliFromLogOddsOp.LogOddsAverageLogarithm(true, x, toLogOddsPrev);
            return Gaussian.FromNatural(scale * toLogOdds.MeanTimesPrecision + shift, scale * toLogOdds.Precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogisticOp"]/message_doc[@name="XAverageLogarithm(double)"]/*'/>
        public static Gaussian XAverageLogarithm(double logistic)
        {
            return XAverageConditional(logistic);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogisticOp_JJ96"]/doc/*'/>
    /// <remarks>
    /// Uses the Jaakkola and Jordan (1996) bound.
    /// </remarks>
    [FactorMethod(typeof(MMath), "Logistic", typeof(double))]
    [Quality(QualityBand.Preview)]
    public class LogisticOp_JJ96
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogisticOp_JJ96"]/message_doc[@name="AverageLogFactor(Beta, Gaussian, Beta)"]/*'/>
        public static double AverageLogFactor(Beta logistic, [Proper, SkipIfUniform] Gaussian x, Beta to_logistic)
        {
            double a = logistic.TrueCount;
            double b = logistic.FalseCount;
            double scale = a + b - 2;
            double shift = -(b - 1);
            // sigma(x) >= sigma(t) exp((x-t)/2 - a/2*(x^2 - t^2))
            double m, v;
            x.GetMeanAndVariance(out m, out v);
            double t = Math.Sqrt(m * m + v);
            double lambda = (t == 0) ? 0.25 : Math.Tanh(t / 2) / (2 * t);
            double boundOnLogSigma = MMath.LogisticLn(t) + (m - t) / 2.0 - .5 * lambda * (m * m + v - t * t);
            return scale * boundOnLogSigma + shift * m - logistic.GetLogNormalizer() - to_logistic.GetAverageLog(logistic);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogisticOp_JJ96"]/message_doc[@name="LogisticAverageLogarithm(Gaussian)"]/*'/>
        public static Beta LogisticAverageLogarithm([Proper] Gaussian x)
        {
            double m, v;
            x.GetMeanAndVariance(out m, out v);

            // for consistency with XAverageLogarithm
            double t = Math.Sqrt(m * m + v);
            double s = -1;
            double eLogOneMinusP = MMath.LogisticLn(t) + (s * m - t) / 2;
            // E[log (1-sigma(x))] = E[log sigma(-x)] = -E[log(1+exp(x))]
            // E[log sigma(x)] = -E[log(1+exp(-x))] = -E[log(1+exp(x))-x] = -E[log(1+exp(x))] + E[x]
            double eLogP = eLogOneMinusP + m;
            return Beta.FromMeanLogs(eLogP, eLogOneMinusP);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogisticOp_JJ96"]/message_doc[@name="XAverageLogarithm(Beta, Gaussian, Gaussian)"]/*'/>
        public static Gaussian XAverageLogarithm([SkipIfUniform] Beta logistic, [Proper, SkipIfUniform] Gaussian x, Gaussian result)
        {
            if (logistic.IsPointMass)
                return LogisticOp.XAverageLogarithm(logistic.Point);
            // f(x) = sigma(x)^(a-1) sigma(-x)^(b-1)
            //      = sigma(x)^(a+b-2) exp(-x(b-1))
            // since sigma(-x) = sigma(x) exp(-x)

            double a = logistic.TrueCount;
            double b = logistic.FalseCount;
            double scale = a + b - 2;
            if (scale == 0.0)
                return Gaussian.Uniform();
            double shift = -(b - 1);
            // sigma(x) >= sigma(t) exp((x-t)/2 - a/2*(x^2 - t^2))
            double m, v;
            x.GetMeanAndVariance(out m, out v);
            double t = Math.Sqrt(m * m + v);
            double lambda = (t == 0) ? 0.25 : Math.Tanh(t / 2) / (2 * t);
            return Gaussian.FromNatural(scale * 0.5 + shift, scale * lambda);
        }
    }


    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogisticOp_SJ99"]/doc/*'/>
    /// <remarks>
    /// Uses the Saul and Jordan (1999) bound
    /// \langle log(1+exp(x)) \rangle \leq a^2*v/2 + log(1+exp(m+(1-2a)v/2))
    /// </remarks>
    [FactorMethod(typeof(MMath), "Logistic", typeof(double), Default = true)]
    [Quality(QualityBand.Preview)]
    [Buffers("A")]
    public class LogisticOp_SJ99
    {
        public static double global_step = 0.5;

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogisticOp_SJ99"]/message_doc[@name="AverageLogFactor(Beta, Gaussian, Beta, double)"]/*'/>
        public static double AverageLogFactor(Beta logistic, [Proper, SkipIfUniform] Gaussian x, Beta to_logistic, double a)
        {
            double b = logistic.FalseCount;
            double scale = logistic.TrueCount + b - 2;
            double shift = -(b - 1);
            double m, v;
            x.GetMeanAndVariance(out m, out v);
            double boundOnLog1PlusExp = a * a * v / 2.0 + MMath.Log1PlusExp(m + (1.0 - 2.0 * a) * v / 2.0);
            double boundOnLogSigma = m - boundOnLog1PlusExp;
            return scale * boundOnLogSigma + shift * m - logistic.GetLogNormalizer() - to_logistic.GetAverageLog(logistic);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogisticOp_SJ99"]/message_doc[@name="AInit()"]/*'/>
        public static double AInit()
        {
            return .5;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogisticOp_SJ99"]/message_doc[@name="A(Gaussian, double)"]/*'/>
        [NoTriggers] // see VmpTests.AdditiveSparseBlockmodel
        public static double A([Proper, SkipIfUniform] Gaussian x, double a)
        {
            double m, v;
            x.GetMeanAndVariance(out m, out v);
            double newa = .5 - (MMath.Logit(a) - m) / v;
            double normala = MMath.Logistic(m + (1.0 - 2.0 * a) * v / 2.0);
            if (logSumExpBound(m, v, normala) > logSumExpBound(m, v, newa))
                return newa;
            else
                return normala;
        }

        public static double logSumExpBound(double m, double v, double a)
        {
            return 0.5 * v * a * a + MMath.Log1PlusExp(m + (0.5 - a) * v);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogisticOp_SJ99"]/message_doc[@name="LogisticAverageLogarithm(Gaussian, double)"]/*'/>
        public static Beta LogisticAverageLogarithm([Proper] Gaussian x, double a)
        {
            double m, v;
            x.GetMeanAndVariance(out m, out v);

            double eLogOneMinusP = -logSumExpBound(m, v, a);
            // E[log (1-sigma(x))] = E[log sigma(-x)] = -E[log(1+exp(x))]
            // E[log sigma(x)] = -E[log(1+exp(-x))] = -E[log(1+exp(x))-x] = -E[log(1+exp(x))] + E[x]
            double eLogP = eLogOneMinusP + m;
            return Beta.FromMeanLogs(eLogP, eLogOneMinusP);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogisticOp_SJ99"]/message_doc[@name="LogisticInit()"]/*'/>
        [Skip]
        public static Beta LogisticInit()
        {
            return Beta.Uniform();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogisticOp_SJ99"]/message_doc[@name="XInit()"]/*'/>
        [Skip]
        public static Gaussian XInit()
        {
            return Gaussian.Uniform();
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 429
#endif

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogisticOp_SJ99"]/message_doc[@name="XAverageLogarithm(Beta, Gaussian, Gaussian, double)"]/*'/>
        public static Gaussian XAverageLogarithm([SkipIfUniform] Beta logistic, /*[Proper, SkipIfUniform]*/ Gaussian x, Gaussian to_x, double a)
        {
            if (logistic.IsPointMass)
                return LogisticOp.XAverageLogarithm(logistic.Point);
            // f(x) = sigma(x)^(a-1) sigma(-x)^(b-1)
            //      = sigma(x)^(a+b-2) exp(-x(b-1))
            // since sigma(-x) = sigma(x) exp(-x)

            double scale = logistic.TrueCount + logistic.FalseCount - 2;
            if (scale == 0.0)
                return Gaussian.Uniform();
            double shift = -(logistic.FalseCount - 1);
            double m, v;
            x.GetMeanAndVariance(out m, out v);
            double sa;
            if (double.IsPositiveInfinity(v))
            {
                a = 0.5;
                sa = MMath.Logistic(m);
            }
            else
            {
                sa = MMath.Logistic(m + (1 - 2 * a) * v * 0.5);
            }
            double precision = a * a + (1 - 2 * a) * sa;
            // meanTimesPrecision = m*a*a + 1-2*a*sa;
            double meanTimesPrecision = 1 - sa;
            if (precision != 0) meanTimesPrecision += m * precision; // avoid 0 * infinity
            //double vf = 1/(a*a + (1-2*a)*sa);
            //double mf = m + vf*(true ? 1-sa : sa);
            //double precision = 1/vf;
            //double meanTimesPrecision = mf*precision;
            Gaussian result = Gaussian.FromNatural(scale * meanTimesPrecision + shift, scale * precision);
            double step = (LogisticOp_SJ99.global_step == 0.0) ? 1.0 : (Rand.Double() * LogisticOp_SJ99.global_step);
            // random damping helps convergence, especially with parallel updates
            if (false && !x.IsPointMass)
            {
                // if the update would change the sign of 1-2*sa, send a message to make sa=0.5
                double newPrec = x.Precision - to_x.Precision + result.Precision;
                double newv = 1 / newPrec;
                double newm = newv * (x.MeanTimesPrecision - to_x.MeanTimesPrecision + result.MeanTimesPrecision);
                double newarg = newm + (1 - 2 * a) * newv * 0.5;
                if ((sa < 0.5 && newarg > 0) || (sa > 0.5 && newarg < 0))
                {
                    // send a message to make newarg=0
                    // it is sufficient to make (x.MeanTimesPrecision + step*(result.MeanTimesPrecision - to_x.MeanTimesPrecision) + 0.5-a) = 0
                    double mpOffset = x.MeanTimesPrecision + 0.5 - a;
                    double precOffset = x.Precision;
                    double mpScale = result.MeanTimesPrecision - to_x.MeanTimesPrecision;
                    double precScale = result.Precision - to_x.Precision;
                    double arg = m + (1 - 2 * a) * v * 0.5;
                    //arg = 0;
                    step = (arg * precOffset - mpOffset) / (mpScale - arg * precScale);
                    //step = (a-0.5-x.MeanTimesPrecision)/(result.MeanTimesPrecision - to_x.MeanTimesPrecision);
                    //Console.WriteLine(step);
                }
            }
            if (step != 1.0)
            {
                result.Precision = step * result.Precision + (1 - step) * to_x.Precision;
                result.MeanTimesPrecision = step * result.MeanTimesPrecision + (1 - step) * to_x.MeanTimesPrecision;
            }
            return result;
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 429
#endif

    }
}
