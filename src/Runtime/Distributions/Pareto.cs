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
    /// A Pareto distribution over the real numbers from lowerBound to infinity.
    /// </summary>
    /// <remarks>
    /// The distribution has two parameters (shape, lowerBound) which must be positive for the distribution to be proper.  
    /// Its density function is
    /// <c>p(x) = s L^s / x^(s+1)</c> for <c>x &gt;= L</c>.
    /// If shape = -1, the distribution is uniform.
    /// If shape = infinity, the distribution is a point mass on the lowerBound.
    /// </remarks>
    [Serializable]
    [DataContract]
    [Quality(QualityBand.Preview)]
    public struct Pareto : IDistribution<double>, SettableTo<Pareto>, SettableToProduct<Pareto>, Sampleable<double>,
        SettableToRatio<Pareto>, SettableToPower<Pareto>, SettableToWeightedSum<Pareto>, 
        CanGetLogAverageOf<Pareto>, CanGetLogAverageOfPower<Pareto>, CanGetAverageLog<Pareto>
    {
        /// <summary>
        /// The shape parameter
        /// </summary>
        [DataMember]
        public double Shape;

        /// <summary>
        /// The lower bound
        /// </summary>
        [DataMember]
        public double LowerBound;

        /// <summary>
        /// Creates a Pareto distribution with the given parameters.
        /// </summary>
        /// <param name="lowerBound"></param>
        /// <param name="shape"></param>
        [Construction("Shape", "LowerBound")]
        public Pareto(double shape, double lowerBound)
        {
            this.Shape = shape;
            this.LowerBound = lowerBound;
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that"></param>
        public Pareto(Pareto that)
            : this(that.Shape, that.LowerBound)
        {
        }

        /// <summary>
        /// Creates a uniform Pareto distribution.
        /// </summary>
        /// <returns></returns>
        public static Pareto Uniform()
        {
            return new Pareto(0, -1);
        }

        public object Clone()
        {
            return new Pareto(this);
        }

        public void SetTo(Pareto that)
        {
            this.Shape = that.Shape;
            this.LowerBound = that.LowerBound;
        }

        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public double Point
        {
            get
            {
                return LowerBound;
            }
            set
            {
                LowerBound = value;
                Shape = double.PositiveInfinity;
            }
        }

        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public bool IsPointMass
        {
            get
            {
                return double.IsPositiveInfinity(Shape);
            }
        }

        public double MaxDiff(object that)
        {
            if(!(that is Pareto)) return double.PositiveInfinity;
            Pareto thatd = (Pareto)that;
            return Math.Max(Math.Abs(this.LowerBound - thatd.LowerBound), Math.Abs(this.Shape - thatd.Shape));
        }

        public void SetToUniform()
        {
            LowerBound = 0;
            Shape = -1;
        }

        public bool IsUniform()
        {
            return (Shape == -1) && (LowerBound == 0);
        }

        public double GetLogProb(double value)
        {
            if (value < LowerBound)
                return double.NegativeInfinity;
            else
                return -(Shape+1)*Math.Log(value) - GetLogNormalizer();
        }

        public double GetLogNormalizer()
        {
            if (!IsProper())
                return 0.0;
            return -Math.Log(Shape) - Shape * Math.Log(LowerBound);
        }

        public bool IsProper()
        {
            return (LowerBound > 0) && (Shape > 0);
        }

        /// <summary>
        /// Static product operator. Create a Pareto distribution which is the product of
        /// two Pareto distributions
        /// </summary>
        /// <param name="a">The first distribution</param>
        /// <param name="b">The second distribution</param>
        /// <returns>The resulting distribution</returns>
        public static Pareto operator *(Pareto a, Pareto b)
        {
            Pareto result = new Pareto();
            result.SetToProduct(a, b);
            return result;
        }

        public void SetToProduct(Pareto a, Pareto b)
        {
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
                this.LowerBound = Math.Max(a.LowerBound, b.LowerBound);
                this.Shape = a.Shape + b.Shape + 1;
            }            
        }

        public double Sample()
        {
            return Math.Pow(Rand.Double(), -1 / Shape) * LowerBound;
        }

        public double Sample(double result)
        {
            return Sample();
        }

        public void SetToRatio(Pareto numerator, Pareto denominator, bool forceProper = false)
        {
            if(string.Empty.Length==0)
                throw new NotSupportedException();
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
            else if (denominator.IsPointMass || denominator.LowerBound > numerator.LowerBound)
            {
                throw new DivideByZeroException();
            }
            else
            {
                this.LowerBound = numerator.LowerBound;
                this.Shape = numerator.Shape - denominator.Shape - 1;
            }
        }

        public void SetToPower(Pareto value, double exponent)
        {
            throw new NotImplementedException();
        }

        public void SetToSum(double weight1, Pareto value1, double weight2, Pareto value2)
        {
            throw new NotImplementedException();
        }

        public double GetLogAverageOf(Pareto that)
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
                Pareto product = this * that;
                return product.GetLogNormalizer() - this.GetLogNormalizer() - that.GetLogNormalizer();
            }
        }

        public double GetLogAverageOfPower(Pareto that, double power)
        {
            throw new NotImplementedException();
        }

        public double GetAverageLog(Pareto that)
        {
            throw new NotImplementedException();
        }

        public override bool Equals(object that)
        {
            return MaxDiff(that) == 0.0;
        }

        public override int GetHashCode()
        {
            return Hash.Combine(Shape.GetHashCode(), LowerBound.GetHashCode());
        }

        public override string ToString()
        {
            string format = "g4";
            return string.Format("Pareto({0},{1})", Shape.ToString(format), LowerBound.ToString(format));
        }
    }
}
