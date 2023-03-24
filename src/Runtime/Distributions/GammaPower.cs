// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Diagnostics;
    using System.Runtime.Serialization;

    using Factors.Attributes;

    using Math;

    using Microsoft.ML.Probabilistic.Serialization;

    using Utilities;

    /// <summary>
    /// The distribution of a Gamma variable raised to a power.  Also known as the Generalized Gamma distribution.  The Weibull distribution is a special case.
    /// </summary>
    /// <remarks><para>
    /// The Gamma-power distribution is defined as the distribution of a Gamma(a,b) variable raised to the power c.  
    /// Thus it has three parameters (a,b,c) and probability density function
    /// <c>p(x) = x^(a/c-1)*exp(-b*x^(1/c))*b^a/Gamma(a)/abs(c)</c>.
    /// In this implementation, the <c>a</c> parameter is called the "Shape", the <c>b</c> parameter
    /// is called the "Rate", and the <c>c</c> parameter is called the "Power".  
    /// The power can be any real number, positive or negative.
    /// The distribution is sometimes also parameterized by (shape,scale,power) where scale = 1/rate.
    /// </para><para>
    /// Special cases:
    /// When the power is 1 it reduces to a Gamma distribution.
    /// When the shape is 1 it reduces to a Weibull distribution.
    /// When the shape equals the power and rate is 0, the distribution is uniform.
    /// When the shape is infinity, the distribution is a point mass and the density is delta(x-Point)
    /// where the Point property gives the mean.
    /// When the power is zero, the distribution is a point mass at one.
    /// When a &lt;= 0 or b &lt;= 0 the <c>b^a/Gamma(a)/abs(c)</c> term is dropped.
    /// </para></remarks>
    [Serializable]
    [DataContract]
    [Quality(QualityBand.Experimental)]
    public struct GammaPower : IDistribution<double>,
                               SettableTo<GammaPower>, SettableToProduct<GammaPower>, Diffable, SettableToUniform,
                               SettableToRatio<GammaPower>, SettableToPower<GammaPower>, SettableToWeightedSum<GammaPower>,
                               Sampleable<double>, CanGetMean<double>, CanGetVariance<double>,
                               CanGetMeanAndVarianceOut<double, double>, CanSetMeanAndVariance<double, double>,
                               CanGetLogAverageOf<GammaPower>, CanGetLogAverageOfPower<GammaPower>,
                               CanGetAverageLog<GammaPower>, CanGetLogNormalizer, CanGetMode<double>
    {
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
        /// Power parameter for the distribution
        /// </summary>
        [DataMember]
        public double Power;

        /// <summary>
        /// Get the expected value E(x)
        /// </summary>
        /// <returns>E(x)</returns>
        public double GetMean()
        {
            return GetMeanPower(1);
        }

        /// <summary>
        /// Get the variance
        /// </summary>
        /// <returns>Variance</returns>
        public double GetVariance()
        {
            GetMeanAndVariance(out double mean, out double variance);
            return variance;
        }

        /// <summary>
        /// Get the mean and variance
        /// </summary>
        /// <param name="mean">Where to put the mean</param>
        /// <param name="variance">Where to put the variance</param>
        public void GetMeanAndVariance(out double mean, out double variance)
        {
            if (Power == 0)
            {
                mean = 1;
                variance = 0;
            }
            if (IsPointMass)
            {
                mean = Point;
                variance = 0;
            }
            else if (!IsProper() && Rate != 0)
            {
                throw new ImproperDistributionException(this);
            }
            else
            {
                double logMean = GetLogMeanPower(1);
                mean = Math.Exp(logMean);
                if (Rate == 0 && Shape + 2 * Power == 0 && 2 * Power < -1) variance = 0;
                else if (Shape + 2 * Power <= 0) variance = double.PositiveInfinity;
                else
                {
                    double logMean2 = GetLogMeanPower(2);
                    if (logMean2 > double.MaxValue) variance = double.PositiveInfinity;
                    else variance = MMath.DifferenceOfExp(logMean2, 2 * logMean);
                }
            }
        }

        /// <summary>
        /// Set the mean and variance
        /// </summary>
        /// <param name="mean">Mean</param>
        /// <param name="variance">Variance</param>
        public void SetMeanAndVariance(double mean, double variance)
        {
            if (double.IsNaN(mean)) throw new ArgumentOutOfRangeException(nameof(mean), mean, "mean is NaN");
            if (double.IsNaN(variance)) throw new ArgumentOutOfRangeException(nameof(variance), variance, "variance is NaN");
            if (variance == 0)
            {
                Point = mean;
            }
            else if (variance < 0)
            {
                throw new ArgumentException("variance < 0 (" + variance + ")");
            }
            else if (Power == 1)
            {
                // mean = a/b
                // variance = a/b^2
                Rate = mean / variance;
                Shape = mean * Rate;
            }
            else if (Power == -1)
            {
                // mean = b/(a-1)
                // variance = b^2/(a-1)^2/(a-2)
                double mean2 = mean * mean;
                double r = double.IsPositiveInfinity(mean2) ? 1.0 : mean2 / variance;
                Shape = r + 2;
                Rate = mean * (Shape - 1);
                if (double.IsNaN(Shape)) throw new Exception();
            }
            else if (Power == 2)
            {
                // mean = a*(a+1)/b^2
                // variance = a*(a+1)*(4*a+6)/b^4
                double mean2 = mean * mean;
                double r = double.IsPositiveInfinity(mean2) ? 1.0 : mean2 / variance;
                double b = 2 * r - 0.5;
                Shape = Math.Sqrt(b * b + 6 * r) + b;
                Rate = Math.Sqrt(Shape * (Shape + 1) / mean);
            }
            else
            {
                throw new NotImplementedException("Not implemented for power " + Power);
            }
            CheckForPointMass();
        }

        /// <summary>
        /// Creates a new Gamma distribution from mean and variance
        /// </summary>
        /// <param name="mean">Mean</param>
        /// <param name="variance">Variance</param>
        /// <param name="power"></param>
        /// <returns>A new Gamma instance</returns>
        public static GammaPower FromMeanAndVariance(double mean, double variance, double power)
        {
            GammaPower g = new GammaPower();
            g.Power = power;
            g.SetMeanAndVariance(mean, variance);
            return g;
        }

        /// <summary>
        /// Sets the shape and rate (rate = 1/scale) parameters of the distribution.
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="rate">rate = 1/scale</param>
        /// <param name="power"></param>
        public void SetShapeAndRate(double shape, double rate, double power)
        {
            this.Power = power;
            this.Shape = shape;
            this.Rate = rate;
            CheckForPointMass();
        }

        private void CheckForPointMass()
        {
            if (!IsPointMass && Rate > double.MaxValue)
            {
                Point = Math.Pow(0, Power);
            }
        }

        /// <summary>
        /// Constructs a Gamma distribution with the given shape and rate parameters.
        /// </summary>
        /// <param name="shape">shape</param>
        /// <param name="rate">rate = 1/scale</param>
        /// <param name="power"></param>
        /// <returns>A new Gamma distribution.</returns>
        [Construction("Shape", "Rate", "Power")]
        public static GammaPower FromShapeAndRate(double shape, double rate, double power)
        {
            GammaPower result = new GammaPower();
            result.SetShapeAndRate(shape, rate, power);
            return result;
        }

        /// <summary>
        /// Sets the shape and scale for this instance
        /// </summary>
        /// <param name="shape">Shape</param>
        /// <param name="scale">Scale</param>
        /// <param name="power"></param>
        public void SetShapeAndScale(double shape, double scale, double power)
        {
            this.Power = power;
            this.Shape = shape;
            this.Rate = 1.0 / scale;
            CheckForPointMass();
        }

        /// <summary>
        /// Constructs a GammaPower distribution with the given shape and scale parameters.
        /// </summary>
        /// <param name="shape">shape</param>
        /// <param name="scale">scale</param>
        /// <param name="power"></param>
        /// <returns>A new Gamma distribution.</returns>
        public static GammaPower FromShapeAndScale(double shape, double scale, double power)
        {
            GammaPower result = new GammaPower();
            result.SetShapeAndScale(shape, scale, power);
            return result;
        }

        /// <summary>
        /// Constructs a GammaPower distribution with the given mean and mean logarithm.
        /// </summary>
        /// <param name="mean">Desired expected value.</param>
        /// <param name="meanLog">Desired expected logarithm.</param>
        /// <param name="power">Desired power.</param>
        /// <returns>A new GammaPower distribution.</returns>
        /// <remarks>This function is equivalent to maximum-likelihood estimation of a Gamma distribution
        /// from data given by sufficient statistics.
        /// This function is significantly slower than the other constructors since it
        /// involves nonlinear optimization. The algorithm is a generalized Newton iteration, 
        /// described in "Estimating a Gamma distribution" by T. Minka, 2002.
        /// </remarks>
        public static GammaPower FromMeanAndMeanLog(double mean, double meanLog, double power)
        {
            return FromLogMeanMinusMeanLog(mean, Math.Log(mean) - meanLog, power);
        }

        /// <summary>
        /// Constructs a GammaPower distribution with the given mean and mean logarithm.
        /// </summary>
        /// <param name="mean">Desired expected value.</param>
        /// <param name="logMeanMinusMeanLog">Logarithm of desired expected value minus desired expected logarithm.</param>
        /// <param name="power">Desired power.</param>
        /// <returns>A new GammaPower distribution.</returns>
        /// <remarks>This function is equivalent to maximum-likelihood estimation of a Gamma distribution
        /// from data given by sufficient statistics.
        /// This function is significantly slower than the other constructors since it
        /// involves nonlinear optimization. The algorithm is a generalized Newton iteration, 
        /// described in "Estimating a Gamma distribution" by T. Minka, 2002.
        /// </remarks>
        public static GammaPower FromLogMeanMinusMeanLog(double mean, double logMeanMinusMeanLog, double power)
        {
            if (power == 1) return FromGamma(Gamma.FromLogMeanMinusMeanLog(mean, logMeanMinusMeanLog), power);
            if (logMeanMinusMeanLog <= 0)
            {
                // By Jensen's inequality, log(E[x]) >= E[log(x)]
                // with equality only for a point mass.
                return PointMass(mean, power);
            }
            // Constraints:
            // mean = Gamma(Shape + power)/Gamma(Shape)/Rate^power
            // meanLog = power*(digamma(Shape) - log(Rate))
            // log(mean) = log(Gamma(Shape + power)) - log(Gamma(Shape)) - power*log(Rate)
            double logMeanMinusMeanLogOverPower = logMeanMinusMeanLog / power;
            double shape = (power == -1) ? (1 + 0.5 / MMath.ExpMinus1(logMeanMinusMeanLog)) : Math.Max(1 - power, 1);
            int maxiter = 10000;
            int backtrackCount = 0;
            bool previouslyIncreased = false;
            for (int iter = 0; iter < maxiter; iter++)
            {
                double oldShape = shape;
                if (power == -1)
                {
                    // mean = rate/(shape - 1)
                    // meanLog = log(rate) - digamma(shape)
                    // derivative wrt shape is trigamma(shape) =approx 1/(shape - 0.5)
                    double digammaShape = MMath.Digamma(shape);
                    if (logMeanMinusMeanLog > 1 || shape <= 1 || backtrackCount > 5)
                    {
                        // derivative wrt logRate is exp(logRate - logMean) 
                        // = exp(meanLog - logMean + digamma(shape)) 
                        // =approx exp(meanLog - logMean)*(shape - 0.5)
                        // So the convergence rate is approx exp(meanLog - logMean)
                        // If meanLog is close to logMean, this will converge very slowly.
                        //shape = 1 + Math.Exp(logRate - logMean);
                        shape = 1 + Math.Exp(digammaShape - logMeanMinusMeanLog);
                    }
                    else if (backtrackCount > 1)
                    {
                        // digamma(Shape) =approx log(Shape) - 0.5/Shape
                        // meanLog = log(rate) - digamma(shape) + log(shape) -log(shape)
                        // exp(log(mean) - meanLog - digamma(shape) + log(shape))*(shape-1) = shape
                        // (exp(...) - 1)*(shape-1) = 1
                        shape = 1 + 1 / MMath.ExpMinus1(Math.Log(shape) - digammaShape + logMeanMinusMeanLog);
                    }
                    else
                    {
                        // digamma(Shape) =approx log(Shape - 0.5)
                        // shape = 1 + Math.Exp(-delta) * (shape - 0.5);
                        // (1 - exp(-delta)) * shape = 1 - exp(-delta)/2
                        // shape = 1 + exp(-delta)/2/(1 - exp(-delta)) = 1 + 0.5/(exp(delta)-1)
                        // shape = 1 + 0.5 / (Math.Exp(delta) - 1);
                        // This comes from:
                        // meanLog = log(rate) - digamma(shape) + log(shape-0.5)-log(shape-0.5)
                        // = log(mean) + log(shape-1) - digamma(shape) + log(shape-0.5)-log(shape-0.5)
                        shape = 1 + 0.5 / MMath.ExpMinus1(Math.Log(shape - 0.5) - digammaShape + logMeanMinusMeanLog);
                    }
                }
                else
                {
                    // Rate^power = Gamma(Shape + power)/Gamma(Shape)/mean
                    // power*log(Rate) = log(Gamma(Shape + power)) - log(Gamma(Shape)) - log(mean)
                    // derivative wrt shape is (digamma(Shape + power) - digamma(Shape))/power
                    double risingFactorial = MMath.RisingFactorialLnOverN(shape, power);
                    if (power > 1 && backtrackCount < 2)
                    {
                        // logRate = risingFactorial - logMeanOverPower
                        // meanLogOverPower = digamma(shape) - logRate
                        // = digamma(shape) - risingFactorial + logMeanOverPower
                        // = digamma(shape)-log(shape)+0.5/shape +log(shape)+(power-1)*0.5/shape - risingFactorial + logMeanOverPower - power*0.5/shape
                        // power*0.5/shape = digamma(shape)-log(shape)+0.5/shape +log(shape)+(power-1)*0.5/shape - risingFactorial + delta/power
                        // log(x+n) <= log(x) + n/x
                        // risingFactorial <= log(x) + (n-1)/2/x  if n >= 1
                        // risingFactorial <= log(x) + (1-n)/2/x  if n <= -1
                        double logShape = Math.Log(shape);
                        double halfOverShape = 0.5 / shape;
                        shape = power * 0.5 / ((MMath.Digamma(shape) - (logShape - halfOverShape)) + ((power - 1) * halfOverShape + logShape - risingFactorial) + logMeanMinusMeanLogOverPower);
                    }
                    else
                    {
                        // derivative wrt logRate is exp(meanLogOverPower + logRate)
                        // = exp(meanLogOverPower - logMeanOverPower)*Gamma(Shape+power)/Gamma(Shape)
                        // For power=-1 and large shape, the convergence rate is approx 
                        // exp(meanLogOverPower - logMeanOverPower)/(Shape-1)*(log(Shape-0.5) - log(Shape-1.5))
                        // which is small when meanLog is close to logMean.
                        if (shape > 1 && backtrackCount < 2)
                        {
                            // This comes from:
                            // meanLog = power*(digamma(Shape)-log(Shape)+log(Shape) - log(Rate))
                            shape = Math.Exp(Math.Log(shape) - MMath.Digamma(shape) + risingFactorial - logMeanMinusMeanLogOverPower);
                        }
                        else
                        {
                            // This also works but is slower.
                            shape = MMath.DigammaInv(risingFactorial - logMeanMinusMeanLogOverPower);
                        }
                    }
                }
                //Console.WriteLine($"shape = {shape:g17}, logRate = {logRate:g17}, mean = {Math.Exp(power * (MMath.RisingFactorialLnOverN(shape, power) - logRate))} should be {mean}, meanLog = {power * (MMath.Digamma(shape) - logRate)} should be {meanLog}");
                if (MMath.AreEqual(oldShape, shape))
                {
                    //Console.WriteLine($"FromMeanAndMeanLog: {iter + 1} iters");
                    //Console.WriteLine($"shape = {shape:g17}, logRate = {logRate:g17}, mean = {Math.Exp(logRate) / (shape - 1)} should be {mean}, meanLog = {power * (MMath.Digamma(shape) - logRate)} should be {meanLog}");
                    break;
                }
                if (double.IsNaN(shape) || iter == maxiter - 1) throw new Exception("Failed to converge");
                bool increased = (shape > oldShape);
                if (iter > 4 && increased != previouslyIncreased)
                    backtrackCount++;
                previouslyIncreased = increased;
            }
            double rate = (mean > double.MaxValue && power < 0) ? mean : Math.Exp(MMath.RisingFactorialLnOverN(shape, power) - Math.Log(mean) / power);
            return FromShapeAndRate(shape, rate, power);
        }

        /// <summary>
        /// Constructs a GammaPower distribution from a Gamma distribution.
        /// </summary>
        /// <param name="gamma"></param>
        /// <param name="power"></param>
        /// <returns></returns>
        public static GammaPower FromGamma(Gamma gamma, double power)
        {
            return FromShapeAndRate(gamma.Shape, gamma.Rate, power);
        }

        /// <summary>
        /// Computes E[log(x)]
        /// </summary>
        /// <returns></returns>
        public double GetMeanLog()
        {
            if (Power == 0.0) return 0.0;
            else if (IsPointMass) return Math.Log(Point);
            else if (!IsProper()) throw new ImproperDistributionException(this);
            else return Power * (MMath.Digamma(Shape) - Math.Log(Rate));
        }

        /// <summary>
        /// Computes E[x^power]
        /// </summary>
        /// <returns></returns>
        public double GetMeanPower(double power)
        {
            return Math.Exp(GetLogMeanPower(power));
        }

        /// <summary>
        /// Computes log(E[x^power])
        /// </summary>
        /// <returns></returns>
        public double GetLogMeanPower(double power)
        {
            if (power == 0.0 || Power == 0.0) return 0.0;
            else if (IsPointMass) return power * Math.Log(Point);
            //else if (Rate == 0.0) return (Power * power > 0) ? Double.PositiveInfinity : 0.0;
            else if (Shape <= 0 || Rate < 0) throw new ImproperDistributionException(this);
            else
            {
                double power2 = Power * power;
                if (Shape + power2 <= 0)
                {
                    if (Shape + power2 == 0 && Rate == 0)
                    {
                        // Near zero, GammaLn(x) = -log(x)-log(x+1) so if rate=0 we get -(power2+1)*log(0)
                        // Example: GammaPower.FromShapeAndRate(1,0,-1).GetMeanPower(1) is undefined
                        if (power2 == -1) throw new ImproperDistributionException(this);
                        // Example: GammaPower.FromShapeAndRate(0,0,-1).GetMeanPower(1) = 0
                        else if (power2 < -1) return double.NegativeInfinity;
                    }
                    //throw new ArgumentException($"Cannot compute E[x^{power}] since shape ({Shape}) <= {-power2}");
                    return Double.PositiveInfinity;
                }
                else if (power2 > double.MaxValue) return double.PositiveInfinity;
                else
                {
                    // The formula must be computed this way to get consistency between power=1 and power=2 in 32-bit mode.
                    double diff = MMath.RisingFactorialLnOverN(Shape, power2) - Math.Log(Rate);
                    double diffTimesPower = (double)diff * Power;
                    return (double)diffTimesPower * power;
                }
            }
        }

        /// <summary>
        /// Returns x where GetLogProb(x) is highest.
        /// </summary>
        /// <returns></returns>
        public double GetMode()
        {
            if (Power == 0.0) return 1.0;
            else if (IsPointMass) return Point;
            else if (Shape <= Power && Rate == 0) throw new ImproperDistributionException(this);
            else return GetMode(Shape, Rate, Power);
        }

        /// <summary>
        /// Returns x where GetLogProb(x, shape, rate, power) is highest.
        /// </summary>
        /// <param name="shape">Shape parameter</param>
        /// <param name="rate">Rate parameter</param>
        /// <param name="power">Power parameter</param>
        /// <returns></returns>
        public static double GetMode(double shape, double rate, double power)
        {
            // Compute Math.Pow((shape - power)/rate, power) without overflow.
            double shapeMinusPower = Math.Max(0, shape - power);
            bool alwaysUseExp = true;
            if (alwaysUseExp)
            {
                double logMode = Math.Log(shapeMinusPower) - Math.Log(rate);
                return Math.Exp(logMode * power);
            }
            else
            {
                // This approach is more accurate but harder to invert
                double x = shapeMinusPower / rate;
                // The ratio can overflow if rate < 1 and shapeMinusPower > 1e-30
                // The ratio can underflow if rate > 1 and shapeMinusPower < 1
                if ((power >= -1 || power < 1) && (x == 0 || double.IsPositiveInfinity(x)))
                {
                    // x may have overflowed or underflowed.
                    // log(shapeMinusPower) - log(rate) can never overflow or underflow.
                    double logMode = Math.Log(shapeMinusPower) - Math.Log(rate);
                    return Math.Exp(logMode * power);
                }
                else
                {
                    return Math.Pow(x, power);
                }
            }
        }

        /// <summary>
        /// Compute the probability that a sample from this distribution is less than x.
        /// </summary>
        /// <param name="x">Any real number.</param>
        /// <returns>The cumulative gamma distribution at <paramref name="x"/></returns>
        public double GetProbLessThan(double x)
        {
            if (x < 0) return 0;
            if (Power == 0) return (x > 1) ? 1.0 : 0.0;
            double probLessThan = Gamma.FromShapeAndRate(Shape, Rate).GetProbLessThan(Math.Pow(x, 1 / Power));
            if (Power > 0)
            {
                // If power > 0, Pr(gamma^power <= value) = Pr(gamma <= value^(1/power))
                return probLessThan;
            }
            else
            {
                // If power < 0, Pr(gamma^power <= value) = Pr(gamma >= value^(1/power))
                return 1 - probLessThan;
            }
        }

        /// <summary>
        /// Returns the value x such that GetProbLessThan(x) == probability.
        /// </summary>
        /// <param name="probability">A real number in [0,1].</param>
        /// <returns></returns>
        public double GetQuantile(double probability)
        {
            if (Power < 0)
            {
                probability = 1 - probability;
            }
            double quantile = Gamma.FromShapeAndRate(Shape, Rate).GetQuantile(probability);
            return Math.Pow(quantile, Power);
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
        /// Sets/gets the instance as a point mass
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public double Point
        {
            get
            {
                // The accessor must succeed, even if the distribution is not a point mass.
                return Rate;
            }
            set
            {
                // Change this instance to a point mass.
                Shape = Double.PositiveInfinity;
                Rate = value;
            }
        }

        /// <summary>
        /// Set shape and rate to be a uniform distribution, without changing the power.
        /// </summary>
        public void SetToUniform()
        {
            Shape = Power;
            Rate = 0;
        }

        /// <summary>
        /// Asks whether this instance is uniform
        /// </summary>
        /// <returns>True if uniform, false otherwise</returns>
        public bool IsUniform()
        {
            return !IsPointMass && (Shape == Power) && (Rate == 0);
        }

        /// <summary>
        /// Logarithm of the density function.
        /// </summary>
        /// <param name="x">Where to evaluate the density</param>
        /// <param name="shape">Shape parameter</param>
        /// <param name="rate">Rate parameter</param>
        /// <param name="power">Power parameter</param>
        /// <returns>log(GammaPower(x;shape,rate,power))</returns>
        /// <remarks>
        /// The distribution is <c>p(x) = x^(a/c-1)*exp(-b*x^(1/c))*b^a/Gamma(a)/abs(c)</c>.
        /// When a &lt;= 0 or b &lt;= 0 or c = 0 the <c>b^a/Gamma(a)/abs(c)</c> term is dropped.
        /// Thus if shape = 1 and rate = 0 the density is 1.
        /// </remarks>
        public static double GetLogProb(double x, double shape, double rate, double power)
        {
            if (x < 0) return double.NegativeInfinity;
            if (power == 0.0) return (x == 1.0) ? 0.0 : Double.NegativeInfinity;
            if (double.IsPositiveInfinity(x)) // Avoid subtracting infinities below
            {
                if (power > 0)
                {
                    if (rate > 0) return -x;
                    else if (rate < 0) return x;
                    // Fall through when rate == 0
                }
                else // This case avoids inf/inf below
                {
                    if (shape > power) return -x;
                    else if (shape < power) return x;
                    // Fall through when shape == power
                }
            }
            double logRate = Math.Log(rate);
            double logx = Math.Log(x);
            double logShapeMinusPower = Math.Log(Math.Max(0, shape - power));
            double logMode = Math.Log(GetMode(shape, rate, power));  // power * (logShapeMinusPower - logRate);
            if (!double.IsNegativeInfinity(logMode))
            {
                // We compute the density in a way that ensures the maximum is at the mode returned by GetMode.
                // mode = ((shape - power)/rate)^power
                // The part of the log-density that depends on x is:
                //   (shape/power-1)*log(x) - rate*x^(1/power)
                // = (shape/power-1)*(log(x) - power/(shape-power)*rate*x^(1/power))
                // = (shape/power-1)*(log(x) - power*(x/mode)^(1/power))
                // = (shape/power-1)*(log(x/mode) - power*(x/mode)^(1/power) + log(mode))
                // = (shape/power-1)*(log(x/mode) - power*(x/mode)^(1/power)) + (shape-power)*log((shape-power)/rate)
                // = (shape-power)*(log(x/mode)/power - exp(log(x/mode)/power)) + (shape-power)*log((shape-power)/rate)
                // = (shape-power)*(log(x/mode)/power + 1 - exp(log(x/mode)/power)) + (shape-power)*(log((shape-power)/rate) - 1)
                // = -(shape-power)*(ExpMinus1RatioMinus1RatioMinusHalf(log(x/mode)/power) + 0.5)*(log(x/mode)/power)^2) + (shape-power)*(log((shape-power)/rate) - 1)
                double xOverModeLn = logx - logMode;
                double xOverModeLnOverPower = xOverModeLn / power;
                if (Math.Abs(xOverModeLnOverPower) < 10)
                {
                    double result;
                    if (double.IsNegativeInfinity(xOverModeLnOverPower)) result = (shape / power - 1) * logx;
                    else if (double.IsPositiveInfinity(xOverModeLnOverPower) || double.IsPositiveInfinity(MMath.ExpMinus1RatioMinus1RatioMinusHalf(xOverModeLnOverPower))) return (power - shape) * double.PositiveInfinity;
                    else result = -(shape - power) * (MMath.ExpMinus1RatioMinus1RatioMinusHalf(xOverModeLnOverPower) + 0.5) * xOverModeLnOverPower * xOverModeLnOverPower;
                    if (IsProper(shape, rate, power))
                    {
                        // Remaining terms are: 
                        // shape * Math.Log(rate) - MMath.GammaLn(shape) - Math.Log(Math.Abs(power)) + (shape - power) * (Math.Log((shape - power) / rate) - 1)
                        if (shape > 1e10)
                        {
                            //result += power * (1 + Math.Log(rate)) - Math.Log(Math.Abs(power));
                            // In double precision, we can assume GammaLn(x) = (x-0.5)*log(x) - x for x > 1e10
                            //result += (shape - power) * Math.Log(1 - power / shape) + (0.5 - power) * Math.Log(shape);
                            result += shape * Math.Log(1 - power / shape) + power * (1 + logRate - logShapeMinusPower) + 0.5 * Math.Log(shape) - Math.Log(Math.Abs(power));
                        }
                        else
                        {
                            //result += (shape - power) * Math.Log(shape - power) - shape - MMath.GammaLn(shape);
                            result += (shape - power) * (logShapeMinusPower - logRate - 1);
                            result += shape * logRate - MMath.GammaLn(shape) - Math.Log(Math.Abs(power));
                        }
                    }
                    else
                    {
                        result += (shape - power) * (logShapeMinusPower - logRate - 1);
                    }
                    return result;
                }
                // Fall through
            }

            {
                double result = 0;
                double logxOverPower = logx / power;
                if (logx == 0) // avoid inf * 0
                {
                    result -= rate;
                }
                else if (Math.Abs(logxOverPower) < MMath.SqrtUlp1)
                {
                    // The part of the log-density that depends on x is:
                    //   (shape/power-1)*log(x) - rate*x^(1/power)
                    // = (shape/power-1)*log(x) - rate*exp(log(x)/power))
                    //   (when abs(log(x)/power)^2 < ulp(1), exp(log(x)/power) can be approximated by 1 + log(x)/power, because
                    //    the third term of this power series 0.5*log(x)^2/power^2 (as well as all other terms) is less than
                    //    half-ulp of the first)
                    // = (shape/power-1)*log(x) - rate*(1 + log(x)/power)
                    // = ((shape - rate)/power - 1)*log(x) - rate
                    result += ((shape - rate) / power - 1) * logx - rate;
                }
                else
                {
                    if (shape != power && x != 1)
                    {
                        result += (shape / power - 1) * logx;
                    }
                    if (rate != 0)
                    {
                        double xInvPowerRate = -Math.Exp(logxOverPower + logRate);
                        if (double.IsInfinity(xInvPowerRate) && x != 0 && !double.IsPositiveInfinity(x))
                        {
                            // recompute another way to avoid overflow
                            double logRatePower = logRate * power;
                            if (!double.IsInfinity(logRatePower))
                                xInvPowerRate = -Math.Exp((logx + logRatePower) / power);
                        }
                        if (double.IsInfinity(xInvPowerRate)) return xInvPowerRate;
                        result += xInvPowerRate;
                    }
                }
                if (IsProper(shape, rate, power))
                {
                    double minusLogNormalizer;
                    if (shape > 1e10)
                    {
                        double logShape = Math.Log(shape);
                        // In double precision, we can assume GammaLn(x) = (x-0.5)*log(x) - x + MMath.LnSqrt2Pi for x > 1e10
                        minusLogNormalizer = shape * (logRate - logShape + 1) + 0.5 * logShape - MMath.LnSqrt2PI - Math.Log(Math.Abs(power));
                        double result1 = result;
                        result += minusLogNormalizer;
                        if ((double.IsNaN(result) || Math.Abs(result) < Math.Abs(minusLogNormalizer) * 1e-10) && shape != power)
                        {
                            if (double.IsNegativeInfinity(logx))
                            {
                                return result1;
                            }
                            // Recompute a different way to avoid subtracting infinities
                            result = -logx - rate * Math.Exp(logxOverPower) + shape * (logRate - logShape + 1 + logx / power) + 0.5 * logShape - MMath.LnSqrt2PI - Math.Log(Math.Abs(power));
                        }
                    }
                    else
                    {
                        result += shape * logRate - MMath.GammaLn(shape) - Math.Log(Math.Abs(power));
                    }
                }
                return result;
            }
        }

        /// <summary>
        /// Logarithm of the density function.
        /// </summary>
        /// <param name="x">Where to evaluate the density</param>
        /// <returns>log(Gamma(x;shape,rate,power))</returns>
        public double GetLogProb(double x)
        {
            if (Power == 0)
            {
                return (x == 1) ? 0.0 : double.NegativeInfinity;
            }
            else if (IsPointMass)
            {
                return (x == Point) ? 0.0 : Double.NegativeInfinity;
            }
            else if (IsUniform())
            {
                return 0;
            }
            else
            {
                return GetLogProb(x, Shape, Rate, Power);
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
                return MMath.GammaLn(Shape) - Shape * Math.Log(Rate) + Math.Log(Math.Abs(Power));
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
        public double GetAverageLog(GammaPower that)
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
                double invPower = 1.0 / that.Power;
                double result = (that.Shape * invPower - 1) * GetMeanLog() - GetMeanPower(invPower) * that.Rate;
                result -= that.GetLogNormalizer();
                return result;
            }
        }

        /// <summary>
        /// Asks whether this GammaPower instance is proper or not. A GammaPower distribution
        /// is proper only if Shape > 0 and 0 &lt; Rate &lt; infinity and Power is not infinite.
        /// </summary>
        /// <returns>True if proper, false otherwise</returns>
        public bool IsProper()
        {
            return (Shape > 0) && (Rate > 0) && !double.IsPositiveInfinity(Rate) && !double.IsInfinity(Power);
        }

        /// <summary>
        /// Asks whether a GammaPower distribution is proper or not. A GammaPower distribution
        /// is proper only if Shape > 0 and 0 &lt; Rate &lt; infinity and Power is not infinite.
        /// </summary>
        /// <param name="shape">shape parameter</param>
        /// <param name="rate">rate parameter</param>
        /// <param name="power">power parameter</param>
        /// <returns>True if proper, false otherwise</returns>
        public static bool IsProper(double shape, double rate, double power)
        {
            return (shape > 0) && (rate > 0) && !double.IsPositiveInfinity(rate) && !double.IsInfinity(power);
        }

        /// <summary>
        /// The log of the integral of the product of this Gamma and that Gamma
        /// </summary>
        /// <param name="that">That Gamma</param>
        /// <returns>The log inner product</returns>
        public double GetLogAverageOf(GammaPower that)
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
                GammaPower product = this * that;
                //if (!product.IsProper()) throw new ArgumentException("The product is improper");
                return product.GetLogNormalizer() - this.GetLogNormalizer() - that.GetLogNormalizer();
            }
        }

        /// <summary>
        /// Get the integral of this distribution times another distribution raised to a power.
        /// </summary>
        /// <param name="that"></param>
        /// <param name="power"></param>
        /// <returns></returns>
        public double GetLogAverageOfPower(GammaPower that, double power)
        {
            if (IsPointMass)
            {
                return power * that.GetLogProb(Point);
            }
            else if (that.IsPointMass)
            {
                if (power < 0) throw new DivideByZeroException("The exponent is negative and the distribution is a point mass");
                return this.GetLogProb(that.Point);
            }
            else
            {
                var product = this * (that ^ power);
                return product.GetLogNormalizer() - this.GetLogNormalizer() - power * that.GetLogNormalizer();
            }
        }

        /// <summary>
        /// Draw a sample from the distribution
        /// </summary>
        /// <returns>The sample value</returns>
        [Stochastic]
        public double Sample()
        {
            if (IsPointMass)
            {
                return Point;
            }
            else
            {
                return Math.Pow(Rand.Gamma(Shape) / Rate, Power);
            }
        }

        /// <summary>
        /// Samples from this Gamma distribution
        /// </summary>
        /// <param name="result">Ignored</param>
        /// <returns>The sample value</returns>
        [Stochastic]
        public double Sample(double result)
        {
            return Sample();
        }

        /// <summary>
        /// Samples from a Gamma distribution with given shape and scale
        /// </summary>
        /// <param name="shape">shape parameter</param>
        /// <param name="scale">scale parameter</param>
        /// <param name="power"></param>
        /// <returns>The sample value</returns>
        [Stochastic]
        public static double Sample(double shape, double scale, double power)
        {
            return Math.Pow(Rand.Gamma(shape) * scale, power);
        }

        /// <summary>
        /// Samples from a Gamma distribution with given mean and variance
        /// </summary>
        /// <param name="mean">mean parameter</param>
        /// <param name="variance">variance parameter</param>
        /// <param name="power"></param>
        /// <returns>The sample value</returns>
        [Stochastic]
        [ParameterNames("sample", "mean", "variance", "power")]
        public static double SampleFromMeanAndVariance(double mean, double variance, double power)
        {
            return FromMeanAndVariance(mean, variance, power).Sample();
        }

        /// <summary>
        /// Set this distribution to have the same parameter values as that
        /// </summary>
        /// <param name="that">the source parameters</param>
        public void SetTo(GammaPower that)
        {
            Rate = that.Rate;
            Shape = that.Shape;
            Power = that.Power;
        }

        /// <summary>
        /// Set the parameters so that the density function equals the product of two given GammaPower density functions
        /// </summary>
        /// <param name="a">The first distribution</param>
        /// <param name="b">The second distribution.  Must have the same power parameter as <paramref name="a"/></param>
        /// <remarks>
        /// The result may not be proper. No error is thrown in this case.
        /// </remarks>
        public void SetToProduct(GammaPower a, GammaPower b)
        {
            Power = a.Power;
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
            else if (a.Power != b.Power)
            {
                throw new ArgumentException("a.Power (" + a.Power + ") != b.Power (" + b.Power + ")");
            }
            else
            {
                Shape = a.Shape + b.Shape - a.Power;
                Rate = a.Rate + b.Rate;
                CheckForPointMass();
            }
        }

        /// <summary>
        /// Creates a new GammaPower distribution whose density function equals the product of two other GammaPower density functions
        /// </summary>
        /// <param name="a">The first distribution</param>
        /// <param name="b">The second distribution.  Must have the same power parameter as <paramref name="a"/></param>
        /// <returns>Result.  May not be proper.</returns>
        public static GammaPower operator *(GammaPower a, GammaPower b)
        {
            GammaPower result = new GammaPower();
            result.SetToProduct(a, b);
            return result;
        }

        /// <summary>
        /// Sets the parameters so that the density function equals the ratio of two given GammaPower density functions
        /// </summary>
        /// <param name="numerator">The numerator distribution. Can be the same object as this.</param>
        /// <param name="denominator">The denominator distribution.  Must have the same power parameter as <paramref name="numerator"/></param>
        /// <param name="forceProper">If true, the result has shape >= power and rate >= 0</param>
        public void SetToRatio(GammaPower numerator, GammaPower denominator, bool forceProper = false)
        {
            Power = numerator.Power;
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
            else if (numerator.Power != denominator.Power)
            {
                throw new ArgumentException("numerator.Power (" + numerator.Power + ") != denominator.Power (" + denominator.Power + ")");
            }
            else if (forceProper && (numerator.Shape < denominator.Shape || numerator.Rate < denominator.Rate))
            {
                // constraints: shape >= power, rate >= 0, ((shape-power)+denominator.shape)/(rate + denominator.rate) = numerator.shape/numerator.rate
                double m = numerator.Shape / numerator.Rate;
                double shape = m * denominator.Rate - denominator.Shape + Power;
                if (shape < Power)
                {
                    Rate = denominator.Shape / m - denominator.Rate;
                    Shape = Power;
                }
                else
                {
                    Shape = shape;
                    Rate = 0;
                }
            }
            else
            {
                Shape = numerator.Shape - denominator.Shape + Power;
                Rate = numerator.Rate - denominator.Rate;
            }
            CheckForPointMass();
        }

        /// <summary>
        /// Creates a new GammaPower distribution whose density function equals the ratio of two other GammaPower density functions
        /// </summary>
        /// <param name="numerator">The numerator distribution</param>
        /// <param name="denominator">The denominator distribution.  Must have the same power parameter as <paramref name="numerator"/></param>
        /// <returns>Result. May not be proper.</returns>
        public static GammaPower operator /(GammaPower numerator, GammaPower denominator)
        {
            GammaPower result = new GammaPower();
            result.SetToRatio(numerator, denominator);
            return result;
        }

        /// <summary>
        /// Creates a new GammaPower distribution corresponding to dividing every sample by a positive constant.
        /// </summary>
        /// <param name="numerator">The numerator distribution</param>
        /// <param name="denominator">The denominator constant.</param>
        /// <returns>Result. May not be proper.</returns>
        public static GammaPower operator /(GammaPower numerator, double denominator)
        {
            if (denominator <= 0) throw new ArgumentOutOfRangeException(nameof(denominator), denominator, "<= 0");
            return GammaPower.FromShapeAndRate(numerator.Shape, numerator.Rate / denominator, numerator.Power);
        }

        /// <summary>
        /// Sets the parameters to represent the power of a source distribution to some exponent.
        /// </summary>
        /// <param name="dist">The source distribution</param>
        /// <param name="exponent">The exponent</param>
        public void SetToPower(GammaPower dist, double exponent)
        {
            Power = dist.Power;
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
            }
            else
            {
                Shape = (dist.Shape - dist.Power) * exponent + dist.Power;
                Rate = dist.Rate * exponent;
                CheckForPointMass();
            }
        }

        /// <summary>
        /// Raises a distribution to a power.
        /// </summary>
        /// <param name="dist">The distribution.</param>
        /// <param name="exponent">The power to raise to.</param>
        /// <returns><paramref name="dist"/> raised to power <paramref name="exponent"/>.</returns>
        public static GammaPower operator ^(GammaPower dist, double exponent)
        {
            GammaPower result = new GammaPower();
            result.SetToPower(dist, exponent);
            return result;
        }

        /// <summary>
        /// Sets the mean and variance to match a mixture of two GammaPower distributions.
        /// </summary>
        /// <param name="weight1">The first weight</param>
        /// <param name="dist1">The first distribution</param>
        /// <param name="weight2">The second weight</param>
        /// <param name="dist2">The second distribution</param>
        public void SetToSum(double weight1, GammaPower dist1, double weight2, GammaPower dist2)
        {
            this.Power = dist1.Power;
            SetTo(Gaussian.WeightedSum(this, weight1, dist1, weight2, dist2));
        }

        /// <summary>
        /// The maximum difference between the parameters of this distribution and that distribution
        /// </summary>
        /// <param name="thatd">distribution to compare to</param>
        /// <returns>The maximum difference</returns>
        /// <remarks><c>a.MaxDiff(b) == b.MaxDiff(a)</c></remarks>
        public double MaxDiff(object thatd)
        {
            if (!(thatd is GammaPower)) return Double.PositiveInfinity;
            GammaPower that = (GammaPower)thatd;
            double diff1 = MMath.AbsDiff(Rate, that.Rate);
            double diff2 = MMath.AbsDiff(Shape, that.Shape);
            double diff3 = MMath.AbsDiff(Power, that.Power);
            return Math.Max(diff1, Math.Max(diff2, diff3));
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
            int hash = Rate.GetHashCode();
            hash = Hash.Combine(hash, Shape.GetHashCode());
            hash = Hash.Combine(hash, Power.GetHashCode());
            return hash;
        }

        /// <summary>
        /// Equals operator
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static bool operator ==(GammaPower a, GammaPower b)
        {
            return a.Equals(b);
        }

        /// <summary>
        /// Not equals operator
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static bool operator !=(GammaPower a, GammaPower b)
        {
            return !(a == b);
        }

        /// <summary>
        /// Creates a GammaPower distribution with given shape and scale parameters (scale = 1/rate) 
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="scale">scale = 1/rate</param>
        /// <param name="power"></param>
        /// <remarks>
        /// The distribution is <c>p(x) = x^(shape-1)*exp(-x/scale)/(scale^shape * Gamma(shape))</c>.
        /// </remarks>
        public GammaPower(double shape, double scale, double power)
        {
            Shape = shape;
            Rate = 1.0 / scale;
            Power = power;
            CheckForPointMass();
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that"></param>
        public GammaPower(GammaPower that)
        {
            Shape = that.Shape;
            Rate = that.Rate;
            Power = that.Power;
            //SetTo(that);
        }

        /// <summary>
        /// Clones this object. 
        /// </summary>
        /// <returns>An object which is a clone of the current instance. This must be cast
        /// if you want to assign the result to a GammaPower type</returns>
        public object Clone()
        {
            return new GammaPower(this);
        }

        /// <summary>
        /// Create a uniform GammaPower distribution.
        /// </summary>
        /// <returns>A new uniform GammaPower distribution</returns>
        [Construction("Power", UseWhen = "IsUniform"), Skip]
        public static GammaPower Uniform(double power)
        {
            GammaPower result = new GammaPower();
            result.Power = power;
            result.SetToUniform();
            return result;
        }

        /// <summary>
        /// Creates a point mass Gamma distribution
        /// </summary>
        /// <param name="mean">The location of the point mass</param>
        /// <param name="power"></param>
        /// <returns>A new point mass Gamma distribution</returns>
        public static GammaPower PointMass(double mean, double power)
        {
            if (power == 0 && mean != 1) throw new ArgumentOutOfRangeException(nameof(mean), mean, $"mean = {mean} is incompatible with power = 0");
            GammaPower result = new GammaPower();
            result.Power = power;
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
                return "GammaPower.PointMass(" + Point.ToString("g4") + "," + Power.ToString("g4") + ")";
            }
            else if (IsUniform())
            {
                return "GammaPower.Uniform(" + Power.ToString("g4") + ")";
            }
            else
            {
                double scale = 1.0 / Rate;
                string s = "GammaPower(" + Shape.ToString("g4") + ", " + scale.ToString("g4") + ", " + Power.ToString("g4") + ")";
                if (IsProper())
                {
                    s += "[mean=" + GetMean().ToString("g4") + "]";
                }
                return s;
            }
        }
    }
}