// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.Serialization;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Factors.Attributes;
using Microsoft.ML.Probabilistic.Serialization;

namespace Microsoft.ML.Probabilistic.Distributions
{
    /// <summary>
    /// Represents a distribution on a binary variable.
    /// </summary>
    /// <remarks>
    /// The most common way to use the distribution is to get and set its ProbTrue property.
    /// 
    /// The distribution is represented by a single number, the log odds: log p(x=true)/p(x=false).
    /// If this is 0, then the distribution is uniform.
    /// If this is infinity, then the distribution is a point mass on x=true.
    /// If this is -infinity, then the distribution is a point mass on x=false.
    /// In terms of the log odds, p(x=true) = 1/(1+exp(-logOdds)).
    /// </remarks>
    [Serializable]
    [DataContract]
    [Quality(QualityBand.Mature)]
    public struct Bernoulli : IDistribution<bool>,
                              SettableTo<Bernoulli>, SettableToProduct<Bernoulli>, SettableToUniform,
                              SettableToRatio<Bernoulli>, SettableToPower<Bernoulli>, SettableToWeightedSumExact<Bernoulli>,
                              Sampleable<bool>, CanGetMean<double>, CanSetMean<double>, CanGetVariance<double>,
                              CanGetLogAverageOf<Bernoulli>, CanGetLogAverageOfPower<Bernoulli>,
                              CanGetAverageLog<Bernoulli>, CanGetLogNormalizer, CanGetMode<bool>
    {
        /// <summary>
        /// Log odds parameter
        /// </summary>
        [DataMember]
        public double LogOdds;

        /// <summary>
        /// Gets the probability of the binary variable being true
        /// </summary>
        /// <returns>p(x=true)</returns>
        public double GetProbTrue()
        {
            return MMath.Logistic(LogOdds);
        }

        /// <summary>
        /// Sets the probability of the binary variable being true
        /// </summary>
        public void SetProbTrue(double probTrue)
        {
            if (probTrue < 0 || probTrue > 1) throw new ArgumentOutOfRangeException(nameof(probTrue), $"{nameof(probTrue)} = {probTrue} is not in [0,1]");
            LogOdds = MMath.Logit(probTrue);
        }

        /// <summary>
        /// Gets the probability of the binary variable being false
        /// </summary>
        /// <returns>p(x=false)</returns>
        public double GetProbFalse()
        {
            return MMath.Logistic(-LogOdds);
        }

        /// <summary>
        /// Sets the probability of the binary variable being false
        /// </summary>
        public void SetProbFalse(double probFalse)
        {
            if (probFalse < 0 || probFalse > 1) throw new ArgumentOutOfRangeException(nameof(probFalse), $"{nameof(probFalse)} = {probFalse} is not in [0,1]");
            LogOdds = -MMath.Logit(probFalse);
        }

        /// <summary>
        /// Gets the log probability of the binary variable being true
        /// </summary>
        /// <returns>log(p(x=true))</returns>
        public double GetLogProbTrue()
        {
            return MMath.LogisticLn(LogOdds);
        }

        /// <summary>
        /// Sets the log probability of the binary variable being true
        /// </summary>
        public void SetLogProbTrue(double logProbTrue)
        {
            if (logProbTrue > 0) throw new ArgumentOutOfRangeException(nameof(logProbTrue), $"{nameof(logProbTrue)} = {logProbTrue} > 0");
            LogOdds = MMath.LogitFromLog(logProbTrue); //-MMath.LogExpMinus1(-logProbTrue);
        }

        /// <summary>
        /// Gets the log probability of the binary variable being false
        /// </summary>
        /// <returns>log(p(x=false))</returns>
        public double GetLogProbFalse()
        {
            return MMath.LogisticLn(-LogOdds);
        }

        /// <summary>
        /// Sets the log probability of the binary variable being false
        /// </summary>
        public void SetLogProbFalse(double logProbFalse)
        {
            if (logProbFalse > 0) throw new ArgumentOutOfRangeException(nameof(logProbFalse), $"{nameof(logProbFalse)} = {logProbFalse} > 0");
            LogOdds = -MMath.LogitFromLog(logProbFalse); //MMath.LogExpMinus1(-logProbFalse);
        }

        /// <summary>
        /// Whether the distribution is a point mass (true with probability 1 or false
        /// with probability 1)
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public bool IsPointMass
        {
            get { return Double.IsInfinity(LogOdds); }
        }

        /// <summary>
        /// Gets/sets the distribution as a point
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public bool Point
        {
            get { return Double.IsPositiveInfinity(LogOdds); }
            set { LogOdds = value ? Double.PositiveInfinity : Double.NegativeInfinity; }
        }

        /// <summary>
        /// Sets the distribution to uniform
        /// </summary>
        public void SetToUniform()
        {
            LogOdds = 0;
        }

        /// <summary>
        /// Whether the distribution is uniform
        /// </summary>
        /// <returns>True if uniform</returns>
        public bool IsUniform()
        {
            return (LogOdds == 0);
        }

        /// <summary>
        /// Evaluates the logarithm of the density function
        /// </summary>
        /// <param name="x">true or false</param>
        /// <returns>Log of the probability density for the given event</returns>
        public double GetLogProb(bool x)
        {
            return MMath.LogisticLn(x ? LogOdds : -LogOdds);
        }

        /// <summary>
        /// Gets the log normalizer of the distribution
        /// </summary>
        /// <returns>This equals -log(1-p)</returns>
        public double GetLogNormalizer()
        {
            // equivalent to  -log(1-p)
            return MMath.Log1PlusExp(LogOdds);
        }

        /// <summary>
        /// Log of the probability that a draw from this distribution
        /// is equal to a draw from that distribution.
        /// </summary>
        /// <param name="that">That distribution</param>
        /// <returns>The resulting log probability</returns>
        public double GetLogAverageOf(Bernoulli that)
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
                // equivalent to:
                // (this*that).GetLogNormalizer() - this.GetLogNormalizer() - that.GetLogNormalizer()
                return LogProbEqual(LogOdds, that.LogOdds);
            }
        }

        /// <summary>
        /// Get the integral of this distribution times another distribution raised to a power.
        /// </summary>
        /// <param name="that"></param>
        /// <param name="power"></param>
        /// <returns></returns>
        public double GetLogAverageOfPower(Bernoulli that, double power)
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
        /// The expected logarithm of that distribution under this distribution.
        /// </summary>
        /// <param name="that">The distribution to take the logarithm of.</param>
        /// <returns><c>sum_x this.Evaluate(x)*Math.Log(that.Evaluate(x))</c></returns>
        /// <remarks>This is also known as the cross entropy.</remarks>
        public double GetAverageLog(Bernoulli that)
        {
            // result = p*log(q) + (1-p)*log(1-q)
            if (IsPointMass) return Point ? that.GetLogProbTrue() : that.GetLogProbFalse();
            else
            {
                // p must not be 0 or 1.
                double p = GetProbTrue();
                return p * that.GetLogProbTrue() + (1 - p) * that.GetLogProbFalse();
            }
        }

        /// <summary>
        /// Samples from a Bernoulli distribution
        /// </summary>
        /// <returns>The sample value</returns>
        [Stochastic]
        public bool Sample()
        {
            return Bernoulli.Sample(GetProbTrue());
        }

#pragma warning disable 1591
        [Stochastic]
        public bool Sample(bool x)
        {
            return Sample();
        }
#pragma warning restore 1591

        /// <summary>
        /// Samples from a Bernoulli distribution with a specified p(true)
        /// </summary>
        /// <param name="probTrue">p(true)</param>
        /// <returns>The sample</returns>
        [Stochastic]
        public static bool Sample(double probTrue)
        {
            if (probTrue < 0 || probTrue > 1) throw new ArgumentOutOfRangeException(nameof(probTrue), $"{nameof(probTrue)} = {probTrue} is not in [0,1]");
            return (Rand.Double() < probTrue);
        }

        /// <summary>
        /// The most probable value.
        /// </summary>
        /// <returns></returns>
        public bool GetMode()
        {
            return (LogOdds >= 0);
        }

        /// <summary>
        /// Gets the mean of this Bernoulli distribution
        /// </summary>
        /// <returns>The mean</returns>
        public double GetMean()
        {
            return GetProbTrue();
        }

        /// <summary>
        /// Sets the mean of this Bernoulli distribution
        /// </summary>
        /// <param name="mean">The mean</param>
        public void SetMean(double mean)
        {
            SetProbTrue(mean);
        }

        /// <summary>
        /// Gets the variance of this Bernoulli distribution
        /// </summary>
        /// <returns>The variance</returns>
        public double GetVariance()
        {
            if (IsPointMass) return 0.0;
            double p = GetProbTrue();
            return p * (1 - p);
        }

#if class
        public Bernoulli()
        {
            LogOdds = 0.0; // uniform
        }
#endif

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that"></param>
        public Bernoulli(Bernoulli that)
        {
            LogOdds = that.LogOdds;
        }

        /// <summary>
        /// Creates a Bernoulli distribution with given probability of being true.
        /// </summary>
        /// <param name="probTrue">p(true)</param>
        public Bernoulli(double probTrue)
        {
            if (probTrue < 0 || probTrue > 1) throw new ArgumentOutOfRangeException(nameof(probTrue), $"{nameof(probTrue)} = {probTrue} is not in [0,1]");
            LogOdds = MMath.Logit(probTrue);
        }

        /// <summary>
        /// Instantiates a uniform Bernoulli distribution
        /// </summary>
        /// <returns>A new uniform Bernoulli distribution</returns>
        [Construction(UseWhen = "IsUniform"), Skip]
        public static Bernoulli Uniform()
        {
            Bernoulli result = new Bernoulli();
            result.SetToUniform();
            return result;
        }

        /// <summary>
        /// Instantiates a point-mass Bernoulli distribution
        /// </summary>
        /// <param name="value">true of false</param>
        /// <returns>A new point-mass Bernoulli distribution at the specified value</returns>
        public static Bernoulli PointMass(bool value)
        {
            return new Bernoulli(value ? 1.0 : 0.0);
        }

        /// <summary>
        /// Instantiates a Bernoulli distribution from a log-odds value
        /// </summary>
        /// <param name="logOdds">The log-odds</param>
        /// <returns>A new Bernoulli distribution at the specified log-odds</returns>
        [Construction("LogOdds")]
        public static Bernoulli FromLogOdds([SkipIfUniform] double logOdds)
        {
            Bernoulli result = new Bernoulli();
            result.LogOdds = logOdds;
            return result;
        }

        /// <summary>
        /// Sets this instance to a product of two Bernoulli distributions
        /// </summary>
        /// <param name="a">The first distribution</param>
        /// <param name="b">The second distribution</param>
        public void SetToProduct(Bernoulli a, Bernoulli b)
        {
            if (a.IsPointMass && b.IsPointMass && a.Point != b.Point) throw new AllZeroException();
            LogOdds = a.LogOdds + b.LogOdds;
        }

        /// <summary>
        /// Creates a Bernoulli distribution which is the product of two Bernoulli distribution
        /// </summary>
        /// <param name="a">The first distribution</param>
        /// <param name="b">The second distribution</param>
        /// <returns>The resulting Bernoulli distribution</returns>
        public static Bernoulli operator *(Bernoulli a, Bernoulli b)
        {
            Bernoulli result = new Bernoulli();
            result.SetToProduct(a, b);
            return result;
        }

        /// <summary>
        /// Sets this instance to a ratio of two Bernoulli distributions
        /// </summary>
        /// <param name="numerator">The numerator distribution</param>
        /// <param name="denominator">The denominator distribution</param>
        /// <param name="forceProper">Ignored</param>
        public void SetToRatio(Bernoulli numerator, Bernoulli denominator, bool forceProper = false)
        {
            if (numerator.IsPointMass)
            {
                if (denominator.IsPointMass)
                {
                    if (numerator.Point.Equals(denominator.Point))
                    {
                        //SetToUniform();
                        Point = numerator.Point;
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
                LogOdds = numerator.LogOdds - denominator.LogOdds;
            }
        }

        /// <summary>
        /// Creates a Bernoulli distribution which is the ratio of two Bernoulli distribution
        /// </summary>
        /// <param name="numerator">The numerator distribution</param>
        /// <param name="denominator">The denominator distribution</param>
        /// <returns>The resulting Bernoulli distribution</returns>
        public static Bernoulli operator /(Bernoulli numerator, Bernoulli denominator)
        {
            Bernoulli result = new Bernoulli();
            result.SetToRatio(numerator, denominator);
            return result;
        }

        /// <summary>
        /// Sets this instance to the power of a Bernoulli distributions
        /// </summary>
        /// <param name="dist">The distribution to raise to a power</param>
        /// <param name="exponent">The exponent</param>
        public void SetToPower(Bernoulli dist, double exponent)
        {
            if (exponent == 0)
            {
                SetToUniform();
            }
            else
            {
                if (exponent < 0 && dist.IsPointMass)
                    throw new DivideByZeroException($"The {nameof(exponent)} is negative and the distribution is a point mass");
                LogOdds = dist.LogOdds * exponent;
            }
        }

        /// <summary>
        /// Raises this distribution to a power.
        /// </summary>
        /// <param name="dist">The distribution.</param>
        /// <param name="exponent">The power to raise to.</param>
        /// <returns><paramref name="dist"/> raised to power <paramref name="exponent"/>.</returns>
        public static Bernoulli operator ^(Bernoulli dist, double exponent)
        {
            Bernoulli result = new Bernoulli();
            result.SetToPower(dist, exponent);
            return result;
        }

        /// <summary>
        /// Creates the complementary distribution
        /// </summary>
        /// <param name="dist">The original distribution</param>
        /// <returns>The complementary distribution</returns>
        public static Bernoulli operator !(Bernoulli dist)
        {
            return Bernoulli.FromLogOdds(-dist.LogOdds);
        }

        /// <summary>
        /// Creates a Bernoulli distribution which is a weighted sum of two Bernoulli distribution
        /// </summary>
        /// <param name="weight1">The weight for the first distribution</param>
        /// <param name="dist1">The first distribution</param>
        /// <param name="weight2">The weight for the second distribution</param>
        /// <param name="dist2">The second distribution</param>
        /// <returns>The resulting Bernoulli distribution</returns>
        public void SetToSum(double weight1, Bernoulli dist1, double weight2, Bernoulli dist2)
        {
            if (weight1 + weight2 == 0) SetToUniform();
            else if (weight1 + weight2 < 0)
                throw new ArgumentException($"{nameof(weight1)} ({weight1}) + {nameof(weight2)} ({weight2}) < 0");
            else if (weight1 == 0) SetTo(dist2);
            else if (weight2 == 0) SetTo(dist1);
            // if dist1 == dist2 then we must return dist1, with no roundoff error
            else if (dist1.LogOdds == dist2.LogOdds) LogOdds = dist1.LogOdds;
            else if (double.IsPositiveInfinity(weight1))
            {
                if (double.IsPositiveInfinity(weight2))
                {
                    throw new ArgumentException("both weights are infinity");
                }
                else
                {
                    SetTo(dist1);
                }
            }
            else if (double.IsPositiveInfinity(weight2)) SetTo(dist2);
            else
            {
                SetProbTrue((weight1 * dist1.GetProbTrue() + weight2 * dist2.GetProbTrue()) / (weight1 + weight2));
            }
        }

        /// <summary>
        /// Sets this instance to have the parameters of another instance
        /// </summary>
        /// <param name="that">The source Bernoulli distribution</param>
        public void SetTo(Bernoulli that)
        {
            LogOdds = that.LogOdds;
        }

        /// <summary>
        /// Clones this Bernoulli. 
        /// </summary>
        /// <returns>An object which is a clone of the current instance. This must be cast
        /// if you want to assign the result to a Bernoulli type</returns>
        public object Clone()
        {
            Bernoulli that = new Bernoulli();
            that.LogOdds = LogOdds;
            return that;
        }

        /// <summary>
        /// The maximum 'difference' between this instance and that instance.
        /// This returns the absolute difference between the Log-odds
        /// </summary>
        /// <param name="thatd">That distribution</param>
        /// <returns>The resulting maximum difference</returns>
        /// <remarks><c>a.MaxDiff(b) == b.MaxDiff(a)</c></remarks>
        public double MaxDiff(object thatd)
        {
            if (!(thatd is Bernoulli)) return Double.PositiveInfinity;
            Bernoulli that = (Bernoulli)thatd;
            return MMath.AbsDiff(LogOdds, that.LogOdds);
        }

        /// <summary>
        /// Override of the Equals method
        /// </summary>
        /// <param name="thatd">The instance to compare to</param>
        /// <returns>true if the two distributions are the same in value, false otherwise</returns>
        /// <exclude/>
        public override bool Equals(object thatd)
        {
            if (!(thatd is Bernoulli)) return false;
            Bernoulli that = (Bernoulli)thatd;
            return (LogOdds == that.LogOdds);
        }

        /// <summary>
        /// Override of GetHashCode method
        /// </summary>
        /// <returns>The hash code for this instance</returns>
        /// <exclude/>
        public override int GetHashCode()
        {
            return LogOdds.GetHashCode();
        }

        /// <summary>
        /// Equals operator
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static bool operator ==(Bernoulli a, Bernoulli b)
        {
            return a.Equals(b);
        }

        /// <summary>
        /// Not equals operator
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static bool operator !=(Bernoulli a, Bernoulli b)
        {
            return !(a == b);
        }

        /// <summary>
        /// Override of ToString method
        /// </summary>
        /// <returns>String representation of instance - shows distribution type and p(true)</returns>
        /// <exclude/>
        public override string ToString()
        {
            return "Bernoulli(" + GetProbTrue().ToString("g4") + ")";
        }

        /// <summary>
        /// Computes the logical AreEqual function in the log-odds domain.
        /// </summary>
        /// <param name="x">The log-odds of variable A, which can be any real number from -Inf to Inf.</param>
        /// <param name="y">The log-odds of variable B, which can be any real number from -Inf to Inf.</param>
        /// <returns>The log-odds that A == B.</returns>
        /// <remarks>The logical LogitProbEqual function is defined as p1 EQ p2 = p1*p2 + (1-p1)*(1-p2).
        /// It is the same as the complement of XOR: !(p1 XOR p2).
        /// In the log-odds domain, this is:
        /// LogOdds(Logistic(x) EQ Logistic(y)) = log (1+exp(-x-y))/(exp(-x)+exp(-y)).
        /// To compute this reliably when x>0 and y>0, factor out min(x,y).
        /// For other cases, use the identity AE(-x,-y) = AE(x,y) and AE(-x,y) = AE(x,-y) = -AE(x,y).
        /// Note the result is 0 if x=0 or y=0.
        /// </remarks>
        public static double LogitProbEqual(double x, double y)
        {
            double sign = System.Math.Sign(x) * System.Math.Sign(y);
            x = System.Math.Abs(x);
            y = System.Math.Abs(y);
            double sum = x + y, diff = System.Math.Abs(x - y);
            if (x == y) diff = 0; // in case x and y are Inf
            //double min = 0.5 * (sum - diff);  // same as Math.Min(x,y)
            double min = System.Math.Min(x, y);
            double result = min + System.Math.Log((1 + System.Math.Exp(-sum)) / (1 + System.Math.Exp(-diff)));
            return result * sign;
        }

        /// <summary>
        /// Computes the log-probability that A==B where p(A)=Logistic(x), p(B)=Logistic(y).
        /// </summary>
        /// <param name="x">The log-odds of variable A, which can be any real number from -Inf to Inf.</param>
        /// <param name="y">The log-odds of variable B, which can be any real number from -Inf to Inf.</param>
        /// <returns>The log-probability that A==B.</returns>
        /// <remarks>
        /// The result is Math.Log(0.5) if x=0 or y=0.
        /// </remarks>
        public static double LogProbEqual(double x, double y)
        {
            return MMath.LogisticLn(LogitProbEqual(x, y));
        }

        /// <summary>
        /// Computes the logical OR function in the log-odds domain.
        /// </summary>
        /// <param name="x">Any real number, including infinity.</param>
        /// <param name="y">Any real number, including infinity.</param>
        /// <returns>The log odds of (p1 OR p2), where p1 = Logistic(x) and p2 = Logistic(y)</returns>
        /// <remarks>The logical OR function is defined as p1 OR p2 = 1 - (1-p1)*(1-p2) = p1 + p2 - p1*p2.
        /// In the log-odds domain, this is:
        /// LogOdds(Logistic(x) OR Logistic(y)) = log(exp(x) + exp(y) + exp(x+y)).  
        /// To compute this reliably, factor out max(x,y,x+y).
        /// </remarks>
        public static double Or(double x, double y)
        {
            if (x >= 0 && y >= 0)
            {
                return x + y + System.Math.Log(1 + System.Math.Exp(-x) + System.Math.Exp(-y));
            }
            else if (x == y)
            {
                // avoid subtracting infinities
                return x + System.Math.Log(2 + System.Math.Exp(x));
            }
            else if (x > y)
            {
                // y must be negative
                return x + System.Math.Log(1 + System.Math.Exp(y - x) + System.Math.Exp(y));
            }
            else
            {
                // x must be negative
                return y + System.Math.Log(1 + System.Math.Exp(x - y) + System.Math.Exp(x));
            }
        }

        /// <summary>
        /// Computes the Bernoulli gating function in the log-odds domain.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="gate"></param>
        /// <returns></returns>
        /// <remarks>
        /// The Bernoulli gating function is x if gate = -infinity and 0 if gate = infinity.
        /// It is one of the messages sent by a logical OR factor.
        /// In the log-odds domain, this is:
        /// log (1 + exp(-gate))/(1 + exp(-gate-x))
        /// </remarks>
        public static double Gate(double x, double gate)
        {
            if (x == -gate)
            {
                // avoid subtracting infinities
                return -MMath.Ln2 + MMath.Log1PlusExp(-gate);
            }
            else if (gate < 0 && x + gate < 0)
            {
                // factor out -gate and gate+x
                return x + MMath.Log1PlusExp(gate) - MMath.Log1PlusExp(gate + x);
            }
            else
            {
                return MMath.Log1PlusExp(-gate) - MMath.Log1PlusExp(-gate - x);
            }
        }
    }
}