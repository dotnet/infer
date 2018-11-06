// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Runtime.Serialization;

    using Factors.Attributes;
    using Math;
    using Microsoft.ML.Probabilistic.Serialization;
    using Utilities;
    
    /// <summary>
    /// A Gaussian distribution on a periodic domain, such as angles between 0 and 2*pi.
    /// </summary>
    /// <remarks>
    /// The distribution is represented by a unwrapped Gaussian and a period length L.
    /// To get the wrapped density, the unwrapped density is summed over all shifts by L, i.e.
    /// p(x) = sum_k N(x + Lk; m, v)  over all integers k
    /// The wrapped density is automatically normalized over the range [0,L) because it 
    /// simply moves the probability mass that was previously distributed over the real line 
    /// to all lie in the interval [0,L).
    /// </remarks>
    [Serializable]
    [DataContract]
    [Quality(QualityBand.Experimental)]
    public struct WrappedGaussian : IDistribution<double>,
                                    SettableTo<WrappedGaussian>, SettableToProduct<WrappedGaussian>,
                                    SettableToRatio<WrappedGaussian>, SettableToWeightedSum<WrappedGaussian>,
                                    SettableToPower<WrappedGaussian>,
                                    Sampleable<double>,
                                    CanGetLogAverageOf<WrappedGaussian>, CanGetLogAverageOfPower<WrappedGaussian>,
                                    CanGetAverageLog<WrappedGaussian>
    {
        /// <summary>
        /// The unwrapped Gaussian, with mean in [0,L)
        /// </summary>
        [DataMember]
        public Gaussian Gaussian;

        /// <summary>
        /// The period length, greater than zero
        /// </summary>
        [DataMember]
        public double Period;

        /// <summary>
        /// Create a WrappedGaussian distribution
        /// </summary>
        /// <param name="unwrapped"></param>
        /// <param name="period"></param>
        [Construction("Gaussian", "Period")]
        public WrappedGaussian(Gaussian unwrapped, double period)
        {
            this.Gaussian = unwrapped;
            this.Period = period;
        }

        /// <summary>
        /// Create a WrappedGaussian distribution
        /// </summary>
        /// <param name="unwrappedMean"></param>
        /// <param name="unwrappedVariance"></param>
        /// <param name="period"></param>
        public WrappedGaussian(double unwrappedMean, double unwrappedVariance, double period = 2*Math.PI)
        {
            this.Gaussian = new Gaussian(Mod(unwrappedMean, period), unwrappedVariance);
            this.Period = period;
            if (period <= 0) throw new ArgumentException("period (" + period + ") <= 0");
        }

        /// <summary>
        /// Create a uniform WrappedGaussian distribution
        /// </summary>
        /// <param name="period"></param>
        /// <returns></returns>
        [Construction("Period", UseWhen = "IsUniform"), Skip]
        public static WrappedGaussian Uniform(double period = 2*Math.PI)
        {
            WrappedGaussian result = new WrappedGaussian();
            result.Period = period;
            return result;
        }

        /// <summary>
        /// Create a WrappedGaussian distribution with all mass on a single point
        /// </summary>
        /// <param name="point"></param>
        /// <param name="period"></param>
        /// <returns></returns>
        [Construction("Point", "Period", UseWhen = "IsPointMass")]
        public static WrappedGaussian PointMass(double point, double period = 2*Math.PI)
        {
            WrappedGaussian result = new WrappedGaussian();
            result.Period = period;
            result.Point = point;
            return result;
        }

        /// <summary>
        /// Return x modulo the period
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public double Mod(double x)
        {
            return Mod(x, Period);
        }

        /// <summary>
        /// Return x modulo the period
        /// </summary>
        /// <param name="x"></param>
        /// <param name="period"></param>
        /// <returns></returns>
        public static double Mod(double x, double period)
        {
            return x - Math.Floor(x/period)*period;
        }

        /// <summary>
        /// Make a deep copy of this distribution. 
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            WrappedGaussian result = new WrappedGaussian();
            result.SetTo(this);
            return result;
        }

        /// <summary>
        /// Set this distribution to a point mass, or get its location
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public double Point
        {
            get { return Gaussian.Point; }
            set { Gaussian.Point = Mod(value); }
        }

        /// <summary>
        /// True if the distribution is a point mass. 
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public bool IsPointMass
        {
            get { return Gaussian.IsPointMass; }
        }

        /// <summary>
        /// The maximum difference between the parameters of this distribution and that
        /// </summary>
        /// <param name="thatd"></param>
        /// <returns></returns>
        public double MaxDiff(object thatd)
        {
            if (!(thatd is WrappedGaussian)) return Double.PositiveInfinity;
            WrappedGaussian that = (WrappedGaussian) thatd;
            double diff1 = Gaussian.MaxDiff(that.Gaussian);
            double diff2 = Math.Abs(this.Period - that.Period);
            return Math.Max(diff1, diff2);
        }

        /// <summary>
        /// True if this distribution has the same parameters as that
        /// </summary>
        /// <param name="that"></param>
        /// <returns></returns>
        public override bool Equals(object that)
        {
            return (MaxDiff(that) == 0.0);
        }

        /// <summary>
        /// A hash of the distribution parameter values
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            return Hash.Combine(Gaussian.GetHashCode(), Period.GetHashCode());
        }

        /// <summary>
        /// A human-readable string containing the parameters of the distribution
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return "Wrapped" + Gaussian.ToString() + "[period=" + Period.ToString("g4") + "]";
        }

        /// <summary>
        /// Equals operator
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static bool operator ==(WrappedGaussian a, WrappedGaussian b)
        {
            return a.Equals(b);
        }

        /// <summary>
        /// Not equals operator
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static bool operator !=(WrappedGaussian a, WrappedGaussian b)
        {
            return !(a == b);
        }

        /// <summary>
        /// Set the distribution to uniform, keeping the same period
        /// </summary>
        public void SetToUniform()
        {
            Gaussian.SetToUniform();
        }

        /// <summary>
        /// True if the distribution is uniform
        /// </summary>
        /// <returns></returns>
        public bool IsUniform()
        {
            return Gaussian.IsUniform();
        }

        /// <summary>
        /// Get the log probability density at value. 
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public double GetLogProb(double value)
        {
            // p(x) = sum_k N(x + L*k; m, v)
            //      = a/(2*pi) + a/pi sum_{k=1}^inf cos(a*k*(x-m)) exp(-(a*k)^2 v/2) where a = 2*pi/L
            double m, v;
            Gaussian.GetMeanAndVariance(out m, out v);
            if (v < 0.15)
            {
                // use Gaussian summation formula
                double result = double.NegativeInfinity;
                for (int k = -1; k <= 1; k++)
                {
                    double logp = Gaussian.GetLogProb(value + k*Period);
                    result = MMath.LogSumExp(result, logp);
                }
                return result;
            }
            else
            {
                // v >= 0.15
                // use the cosine formula
                double result = 0.5;
                double aOverPi = 2/Period;
                double a = aOverPi*Math.PI;
                double diff = value - m;
                double vHalf = v*0.5;
                for (int k = 1; k <= 8; k++)
                {
                    double ak = a*k;
                    result += Math.Cos(ak*diff)*Math.Exp(-ak*ak*vHalf);
                }
                return Math.Log(result*aOverPi);
            }
        }

        /// <summary>
        /// Set this distribution equal to value
        /// </summary>
        /// <param name="value"></param>
        public void SetTo(WrappedGaussian value)
        {
            this.Gaussian = value.Gaussian;
            this.Period = value.Period;
        }

        /// <summary>
        /// Set this distribution equal to the approximate product of a and b
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <remarks>
        /// Since WrappedGaussians are not closed under multiplication, the result is approximate.
        /// </remarks>
        public void SetToProduct(WrappedGaussian a, WrappedGaussian b)
        {
            if (a.Period < b.Period)
            {
                SetToProduct(b, a);
                return;
            }
            // a.Period >= b.Period
            if (a.IsUniform())
            {
                SetTo(b);
                return;
            }
            if (b.IsUniform())
            {
                SetTo(a);
                return;
            }
            if (a.IsPointMass)
            {
                if (b.IsPointMass && !a.Point.Equals(b.Point)) throw new AllZeroException();
                Point = a.Point;
                return;
            }
            if (b.IsPointMass)
            {
                Point = b.Point;
                return;
            }
            // (a,b) are not uniform or point mass
            double ratio = a.Period/b.Period;
            int intRatio = (int) Math.Round(ratio);
            if (Math.Abs(ratio - intRatio) > a.Period*1e-4) throw new ArgumentException("a.Period (" + a.Period + ") is not a multiple of b.Period (" + b.Period + ")");
            this.Period = a.Period;
            // a.Period = k*b.Period, k >= 1
            // because one period is a multiple of the other, we only need to sum over one set of shifts.
            // otherwise, we would need to sum over two sets of shifts.
            double ma, va, mb, vb;
            a.Gaussian.GetMeanAndVariance(out ma, out va);
            b.Gaussian.GetMeanAndVariance(out mb, out vb);
            double diff = (ma - mb)/b.Period;
#if true
            // approximate using only the one best shift
            int k = (int) Math.Round(diff);
            Gaussian bShifted = new Gaussian(mb + k*b.Period, vb);
            Gaussian.SetToProduct(a.Gaussian, bShifted);
#else
    // we will sum over shifts from kMin to kMax, numbering intRatio in total
            int kMin, kMax;
            if (intRatio % 2 == 1) {
                // odd number of shifts
                int kMid = (int)Math.Round(diff);
                int halfRatio = intRatio / 2;
                kMin = kMid - halfRatio;
                kMax = kMid + halfRatio;
            } else {
                // even number of shifts
                int kMid = (int)Math.Floor(diff);
                int halfRatio = intRatio / 2;
                kMin = kMid - halfRatio + 1;
                kMax = kMid + halfRatio;
            }
            if (kMax - kMin != intRatio-1) throw new InferRuntimeException("kMax - kMin != intRatio-1");
            // exclude shifts that are too far away
            double sa = Math.Sqrt(va + vb);
            double lowerBound = ma - 5*sa;
            double upperBound = ma + 5*sa;
            // find the largest k such that (mb + k*Lb <= lowerBound)
            double kLower = Math.Floor((lowerBound - mb)/b.Period);
            if (kLower > kMin) kMin = (int)kLower;
            // find the smallest k such that (mb + k*Lb >= upperBound)
            double kUpper = Math.Ceiling((upperBound - mb)/b.Period);
            if (kUpper < kMax) kMax = (int)kUpper;
            if (kMax - kMin > 100) throw new InferRuntimeException("kMax - kMin = "+(kMax-kMin));
            double totalWeight = Double.NegativeInfinity;
            for (int k = kMin; k <= kMax; k++) {
                Gaussian bShifted = new Gaussian(mb + k*b.Period, vb);
                Gaussian product = a.Gaussian * bShifted;
                double weight = a.Gaussian.GetLogAverageOf(bShifted);
                if (double.IsNegativeInfinity(totalWeight)) {
                    Gaussian.SetTo(product);
                    totalWeight = weight;
                } else {
                    Gaussian.SetToSum(1.0, Gaussian, Math.Exp(weight - totalWeight), product);
                    totalWeight = MMath.LogSumExp(totalWeight, weight);
                }
            }
#endif
            if (double.IsNaN(Gaussian.MeanTimesPrecision)) throw new InferRuntimeException("result is nan");
            Normalize();
        }

        /// <summary>
        /// Creates a new WrappedGaussian which is the product of two other WrappedGaussians
        /// </summary>
        /// <param name="a">First WrappedGaussian</param>
        /// <param name="b">Second WrappedGaussian</param>
        /// <returns>Result</returns>
        public static WrappedGaussian operator *(WrappedGaussian a, WrappedGaussian b)
        {
            WrappedGaussian result = new WrappedGaussian();
            result.SetToProduct(a, b);
            return result;
        }

        public void Normalize()
        {
            double m, v;
            Gaussian.GetMeanAndVariance(out m, out v);
            Gaussian.SetMeanAndVariance(Mod(m), v);
        }

        /// <summary>
        /// Set this equal to numerator/denominator
        /// </summary>
        /// <param name="numerator"></param>
        /// <param name="denominator"></param>
        /// <param name="forceProper"></param>
        public void SetToRatio(WrappedGaussian numerator, WrappedGaussian denominator, bool forceProper = false)
        {
            throw new NotImplementedException();
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        /// <summary>
        /// Set the parameters to match the moments of a mixture distribution.
        /// </summary>
        /// <param name="value1">The first distribution</param>
        /// <param name="weight1">The first weight</param>
        /// <param name="value2">The second distribution</param>
        /// <param name="weight2">The second weight</param>
        public void SetToSum(double weight1, WrappedGaussian value1, double weight2, WrappedGaussian value2)
        {
            throw new NotImplementedException();
            if (value1.Period != value2.Period) throw new ArgumentException("value1.Period (" + value1.Period + ") != value2.Period (" + value2.Period + ")");
            this.Period = value1.Period;
            this.Gaussian.SetToSum(weight1, value1.Gaussian, weight2, value2.Gaussian);
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

        /// <summary>
        /// Sample from the distribution
        /// </summary>
        /// <returns>The sample value</returns>
        [Stochastic]
        public double Sample()
        {
            return Mod(Gaussian.Sample());
        }

        /// <summary>
        /// Sample from the distribution
        /// </summary>
        /// <param name="result">Ignored</param>
        /// <returns>The sample value</returns>
        [Stochastic]
        public double Sample(double result)
        {
            return Sample();
        }

        /// <summary>
        /// Get the logarithm of the average value of that distribution under this distribution, i.e. log(int this(x) that(x) dx)
        /// </summary>
        /// <param name="that"></param>
        /// <returns></returns>
        public double GetLogAverageOf(WrappedGaussian that)
        {
            // int_L p_wrapped(x) q_wrapped(x) dx = int_x (sum_k p(x + kL)) (sum_j q(x + jL)) dx
            // = sum_kj int_x p(x + kL) q(x + jL) dx
            if (that.Period != this.Period) throw new ArgumentException("that.Period (" + that.Period + ") != this.Period (" + this.Period + ")");
            double m1, v1, m2, v2;
            Gaussian.GetMeanAndVariance(out m1, out v1);
            that.Gaussian.GetMeanAndVariance(out m2, out v2);
            WrappedGaussian prod = new WrappedGaussian(m2, v1 + v2, Period);
            return prod.GetLogProb(m1);
        }

        /// <summary>
        /// Get the integral of this distribution times another distribution raised to a power.
        /// </summary>
        /// <param name="that"></param>
        /// <param name="power"></param>
        /// <returns></returns>
        public double GetLogAverageOfPower(WrappedGaussian that, double power)
        {
            if (IsPointMass)
            {
                return power*that.GetLogProb(Point);
            }
            else if (that.IsPointMass)
            {
                if (power < 0) throw new DivideByZeroException("The exponent is negative and the distribution is a point mass");
                return this.GetLogProb(that.Point);
            }
            else
            {
                throw new NotImplementedException();
            }
        }

        /// <summary>
        /// Get the average logarithm of that distribution under this distribution, i.e. int this(x) log( that(x) ) dx
        /// </summary>
        /// <param name="that"></param>
        /// <returns></returns>
        /// <remarks>
        /// The result is approximate.
        /// </remarks>
        public double GetAverageLog(WrappedGaussian that)
        {
            // int_L p_wrapped(x) log q_wrapped(x) dx = int_x p(x) log q_wrapped(x mod L) dx
            double ma, va, mb, vb;
            this.Gaussian.GetMeanAndVariance(out ma, out va);
            that.Gaussian.GetMeanAndVariance(out mb, out vb);
            if (this.Period < that.Period) throw new ArgumentException("this.Period < that.Period");
            // approximate using the one best shift
            double diff = (ma - mb)/that.Period;
            int k = (int) Math.Round(diff);
            return this.Gaussian.GetAverageLog(new Gaussian(mb + k*that.Period, vb));
        }

        /// <summary>
        /// Set this equal to (dist)^exponent
        /// </summary>
        /// <param name="dist"></param>
        /// <param name="exponent"></param>
        public void SetToPower(WrappedGaussian dist, double exponent)
        {
            throw new NotImplementedException();
        }
    }
}