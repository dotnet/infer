// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Runtime.Serialization;

    using Math;
    using Utilities;
    using Factors.Attributes;
    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// Binomial distribution over the integers [0,n]
    /// </summary>
    /// <remarks>
    /// The formula for the distribution is <c>p(x) = n!/(n-x)!/x! p^x (1-p)^(n-x)</c>.
    /// In this implementation, we use a generalization that includes two extra shape parameters (a,b)
    /// The formula for the generalized distribution is <c>p(x) =propto 1/x!^a 1/(n-x)!^b exp(x*logOdds)</c>.
    /// With this extension, we can represent a uniform distribution via (logOdds=0,a=0,b=0) and
    /// a point mass via logOdds=+/-infinity or a=infinity or b=infinity.
    /// This family is closed under multiplication, while the standard Binomial is not.
    /// </remarks>
    [Serializable]
    [DataContract]
    [Quality(QualityBand.Experimental)]
    public struct Binomial : IDistribution<int>,
                             SettableTo<Binomial>, SettableToProduct<Binomial>, SettableToRatio<Binomial>,
                             SettableToPower<Binomial>, SettableToWeightedSum<Binomial>,
                             CanGetLogAverageOf<Binomial>, CanGetLogAverageOfPower<Binomial>,
                             CanGetAverageLog<Binomial>,
                             Sampleable<int>, CanGetMean<double>, CanGetVariance<double>,
                             CanGetMeanAndVarianceOut<double, double>
    {
        [DataMember]
        public int TrialCount;

        [DataMember]
        public double LogOdds;

        [DataMember]
        public double A;

        [DataMember]
        public double B;

        public double ProbSuccess
        {
            get { return MMath.Logistic(LogOdds); }
        }

        public double ProbFailure
        {
            get { return MMath.Logistic(-LogOdds); }
        }

        public Binomial(int trialCount, double probSuccess)
        {
            TrialCount = trialCount;
            LogOdds = MMath.Logit(probSuccess);
            A = 1;
            B = 1;
        }

        public static Binomial PointMass(int trialCount, int value)
        {
            Binomial result = new Binomial();
            result.TrialCount = trialCount;
            result.Point = value;
            return result;
        }

        [Construction("TrialCount", "LogOdds", "A", "B")]
        public static Binomial FromNatural(int trialCount, double logOdds, double a = 1, double b = 1)
        {
            Binomial result = new Binomial();
            result.TrialCount = trialCount;
            result.LogOdds = logOdds;
            result.A = a;
            result.B = b;
            return result;
        }

        public static Binomial Uniform(int trialCount)
        {
            return Binomial.FromNatural(trialCount, 0, 0, 0);
        }

        public void SetTo(Binomial that)
        {
            this.TrialCount = that.TrialCount;
            this.LogOdds = that.LogOdds;
            this.A = that.A;
            this.B = that.B;
        }

        public double GetMean()
        {
            if (A != 1 || B != 1) throw new NotImplementedException("A != 1 or B !=1 not implemented");
            return TrialCount * ProbSuccess;
        }

        public double GetVariance()
        {
            if (A != 1 || B != 1) throw new NotImplementedException("A != 1 or B !=1 not implemented");
            return TrialCount * ProbSuccess * ProbFailure;
        }

        public void GetMeanAndVariance(out double mean, out double variance)
        {
            if (A != 1 || B != 1) throw new NotImplementedException("A != 1 or B !=1 not implemented");
            mean = TrialCount * ProbSuccess;
            variance = mean * ProbFailure;
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        public void SetToProduct(Binomial a, Binomial b)
        {
            if (a.TrialCount != b.TrialCount) throw new ArgumentException("a.TrialCount (" + a.TrialCount + ") != b.TrialCount (" + b.TrialCount + ")");
            if (a.IsPointMass || b.IsUniform())
            {
                if (b.IsPointMass && !a.Point.Equals(b.Point))
                {
                    throw new AllZeroException();
                }
                SetTo(a);
            }
            else if (b.IsPointMass || a.IsUniform())
            {
                SetTo(b);
            }
            else
            {
                TrialCount = a.TrialCount;
                LogOdds = a.LogOdds + b.LogOdds;
                A = a.A + b.A;
                B = a.B + b.B;
            }
        }

        /// <summary>
        /// Set this distribution to equal the ratio of two distributions
        /// </summary>
        /// <param name="numerator"></param>
        /// <param name="denominator"></param>
        /// <param name="forceProper">Ignored</param>
        public void SetToRatio(Binomial numerator, Binomial denominator, bool forceProper = false)
        {
            if (numerator.TrialCount != denominator.TrialCount)
                throw new ArgumentException("numerator.TrialCount (" + numerator.TrialCount + ") != denominator.TrialCount (" + denominator.TrialCount + ")");
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
                TrialCount = numerator.TrialCount;
                LogOdds = numerator.LogOdds - denominator.LogOdds;
                A = numerator.A - denominator.A;
                B = numerator.B - denominator.B;
            }
        }

        public void SetToPower(Binomial dist, double exponent)
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
                LogOdds = exponent * LogOdds;
                A = exponent * A;
                B = exponent * B;
            }
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

        public void SetToSum(double weight1, Binomial value1, double weight2, Binomial value2)
        {
            throw new NotImplementedException();
        }

        public double GetLogAverageOf(Binomial that)
        {
            throw new NotImplementedException();
        }

        public double GetLogAverageOfPower(Binomial that, double power)
        {
            throw new NotImplementedException();
        }

        public double GetAverageLog(Binomial that)
        {
            throw new NotImplementedException();
        }

        public int Sample()
        {
            if (IsPointMass) return Point;
            if (A != 1 || B != 1) throw new NotImplementedException();
            return Rand.Binomial(TrialCount, ProbSuccess);
        }

        public int Sample(int result)
        {
            return Sample();
        }

        public object Clone()
        {
            Binomial result = new Binomial();
            result.SetTo(this);
            return result;
        }

        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public int Point
        {
            get
            {
                if (double.IsPositiveInfinity(LogOdds))
                {
                    // every trial succeeds
                    return TrialCount;
                }
                else if (double.IsNegativeInfinity(LogOdds))
                {
                    // every trial fails
                    return 0;
                }
                else
                {
                    return (int)Math.Round(MMath.Logistic(LogOdds) * TrialCount);
                }
            }
            set
            {
                A = double.PositiveInfinity;
                B = double.PositiveInfinity;
                LogOdds = MMath.Logit((double)value / TrialCount);
            }
        }

        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public bool IsPointMass
        {
            get { return double.IsInfinity(LogOdds) || double.IsInfinity(A) || double.IsInfinity(B); }
        }

        public double MaxDiff(object that)
        {
            if (!(that is Binomial)) return double.PositiveInfinity;
            Binomial thatd = (Binomial)that;
            if (IsPointMass)
            {
                if (thatd.IsPointMass)
                    return Math.Abs(Point - thatd.Point);
                else return double.PositiveInfinity;
            }
            if (thatd.IsPointMass) return double.PositiveInfinity;
            return Math.Max(Math.Abs(TrialCount - thatd.TrialCount), Math.Abs(LogOdds - thatd.LogOdds));
        }

        public void SetToUniform()
        {
            LogOdds = 0;
            A = 0;
            B = 0;
        }

        public bool IsUniform()
        {
            return (LogOdds == 0) && (A == 0) && (B == 0);
        }

        public double GetLogProb(int value)
        {
            if (value < 0 || value > TrialCount) return double.NegativeInfinity;
            if (IsPointMass) return (value == Point) ? 0.0 : double.NegativeInfinity;
            return MMath.GammaLn(TrialCount + 1) - A * MMath.GammaLn(value + 1) - B * MMath.GammaLn(TrialCount - value + 1) + value * LogOdds + TrialCount * MMath.LogisticLn(-LogOdds);
        }

        public override string ToString()
        {
            if (IsPointMass) return $"Binomial.PointMass({TrialCount},{Point})";
            if (A == 1 && B == 1) return $"Binomial({TrialCount},{ProbSuccess})";
            return $"Binomial.FromNatural({TrialCount},{LogOdds},{A},{B})";
        }
    }
}