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
    /// Represents a discrete distribution in the log domain without explicit normalization.
    /// </summary>
    [Serializable]
    [Quality(QualityBand.Experimental)]
    [DataContract]
    public class UnnormalizedDiscrete : IDistribution<int>, SettableTo<UnnormalizedDiscrete>, Sampleable<int>,
                                        SettableToProduct<UnnormalizedDiscrete>, SettableToRatio<UnnormalizedDiscrete>, SettableToPower<UnnormalizedDiscrete>,
                                        SettableToWeightedSum<UnnormalizedDiscrete>, CanGetLogAverageOf<UnnormalizedDiscrete>, CanGetAverageLog<UnnormalizedDiscrete>
    {
        /// <summary>
        /// Log probability of each value (when not a point mass).  Since the distribution is unnormalized, these may be shifted by an arbitrary constant.
        /// </summary>
        /// <remarks>
        /// logprob.Length == D.
        /// </remarks>
        [DataMember]
        protected DenseVector logProb;

        /// <summary>
        /// Clones this unnormalised discrete distribution. 
        /// </summary>
        /// <returns>An object which is a clone of the current instance. This must be cast
        /// if you want to assign the result to a UnnormalizedDiscrete type</returns>
        public object Clone()
        {
            return FromLogProbs((DenseVector) logProb.Clone());
        }

        /// <summary>
        /// Creates an unnormalized discrete distribution from a vector of log probabilities.
        /// </summary>
        /// <param name="logProb"></param>
        /// <returns></returns>
        public static UnnormalizedDiscrete FromLogProbs(DenseVector logProb)
        {
            UnnormalizedDiscrete ud = new UnnormalizedDiscrete();
            ud.logProb = logProb;
            return ud;
        }

        /// <summary>
        /// Creates an unnormalized discrete distribution from a normal discrete distribution.
        /// </summary>
        /// <param name="d">The discrete distribution</param>
        /// <returns></returns>
        public static UnnormalizedDiscrete FromDiscrete(Discrete d)
        {
            UnnormalizedDiscrete ud = new UnnormalizedDiscrete();
            ud.logProb = (DenseVector) d.GetLogProbs();
            return ud;
        }

        public Discrete ToDiscrete()
        {
            Discrete d = Discrete.Uniform(Dimension);
            d.SetProbs(GetProbs());
            return d;
        }

        /// <summary>
        /// Creates a uniform unnormalized discrete distribution over the values from 0 to numValues-1
        /// </summary>
        /// <returns></returns>
        public static UnnormalizedDiscrete Uniform(int numValues)
        {
            UnnormalizedDiscrete d = new UnnormalizedDiscrete();
            d.logProb = DenseVector.Zero(numValues);
            return d;
        }

        /// <summary>
        /// Dimension of the unnormalised discrete distribution
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public int Dimension
        {
            get { return logProb.Count; }
        }


        /// <summary>
        /// Sets/gets this distribution as a point distribution
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public int Point
        {
            get
            {
                if (Dimension <= 1) return 0;
                else return (int) logProb[1];
            }
            set
            {
                if (Dimension > 1)
                {
                    logProb[0] = Double.PositiveInfinity;
                    logProb[1] = value;
                }
            }
        }

        /// <summary>
        /// Whether or not this instance is a point mass.
        /// </summary>
        [IgnoreDataMember, System.Xml.Serialization.XmlIgnore]
        public bool IsPointMass
        {
            get { return (Dimension <= 1) || Double.IsPositiveInfinity(logProb[0]); }
        }

        #region Diffable Members

        /// <summary>
        /// The maximum difference between the parameters of this discrete and that discrete
        /// </summary>
        /// <param name="that">That discrete</param>
        /// <returns>The maximum difference</returns>
        public double MaxDiff(object that)
        {
            UnnormalizedDiscrete thatd = that as UnnormalizedDiscrete;
            if (thatd == null) return Double.PositiveInfinity;
            return logProb.MaxDiff(thatd.logProb);
        }

        #endregion

        #region SettableToUniform Members

        /// <summary>
        /// Sets this instance to a uniform distribution (i.e. probabilities all equal)
        /// </summary>
        public void SetToUniform()
        {
            logProb.SetAllElementsTo(0);
        }

        /// <summary>
        /// Returns whether the distribution is uniform or not
        /// </summary>
        /// <returns>True if uniform</returns>
        public bool IsUniform()
        {
            return logProb.EqualsAll(logProb[0]);
        }

        #endregion

        #region CanGetLogProb<int> Members

        /// <summary>
        /// Evaluates the log of the unnormalized density at the specified domain value
        /// </summary>
        /// <param name="value">The point at which to evaluate</param>
        /// <returns>The log unnormalized density</returns>
        public double GetLogProb(int value)
        {
            return logProb[value];
        }

        /// <summary>
        /// Evaluates the unnormalized probability at the specified domain value
        /// </summary>
        /// <param name="value">The point at which to evaluate</param>
        /// <returns>The unnormalized probability</returns>
        public double Evaluate(int value)
        {
            if (IsPointMass)
            {
                return (value == Point) ? 1.0 : 0.0;
            }
            else
            {
                return Math.Exp(logProb[value]);
            }
        }

        /// <summary>
        /// Gets a Vector of size this.Dimension.
        /// </summary>
        /// <returns>A pointer to the internal log prob Vector of the object.</returns>
        /// <remarks>
        /// This function is intended to be used with SetLogProbs, to avoid allocating a new Vector.
        /// The return value should not be interpreted as a probs vector, but only a workspace filled
        /// with unknown data that can be overwritten.  Until SetLogProbs is called, the distribution object 
        /// is invalid once this workspace is modified.
        /// </remarks>
        public DenseVector GetWorkspace()
        {
            return logProb;
        }

        #endregion

        /// <summary>
        /// Sets the log probability vector for this distribution.
        /// </summary>
        /// <param name="logProbs">A vector of non-negative, finite numbers.  Need not sum to 1.</param>
        /// <remarks>
        /// Instead of allocating your own Vector to pass to SetProbs, you can call <see cref="GetWorkspace"/>,
        /// fill in the resulting Vector, and then pass it to SetProbs.
        /// </remarks>
        public void SetLogProbs(DenseVector logProbs)
        {
            if (logProbs != logProb)
            {
                logProb.SetTo(logProbs);
            }
        }

        /// <summary>
        /// Sets the parameters of this instance to the parameters of that instance
        /// </summary>
        /// <param name="value">That instance</param>
        public void SetTo(UnnormalizedDiscrete value)
        {
            if (object.ReferenceEquals(value, null)) SetToUniform();
            else
            {
                logProb.SetTo(value.logProb);
            }
        }

        /// <summary>
        /// Sets the parameters to represent the product of two unnormalized discrete distributions.
        /// </summary>
        /// <param name="a">The first unnormalized discrete distribution</param>
        /// <param name="b">The second unnormalized discrete distribution</param>
        public void SetToProduct(UnnormalizedDiscrete a, UnnormalizedDiscrete b)
        {
            if (a.IsPointMass)
            {
                if (b.IsPointMass && (a.Point != b.Point)) throw new AllZeroException();
                SetTo(a);
            }
            else if (b.IsPointMass)
            {
                SetTo(b);
            }
            else
            {
                logProb.SetToSum(a.logProb, b.logProb);
            }
        }

        /// <summary>
        /// Creates an unnormalized discrete distribution which is the product of two unnormalized discrete distributions
        /// </summary>
        /// <param name="a">The first distribution</param>
        /// <param name="b">The second distribution</param>
        /// <returns>The resulting unnormalized discrete distribution</returns>
        public static UnnormalizedDiscrete operator *(UnnormalizedDiscrete a, UnnormalizedDiscrete b)
        {
            UnnormalizedDiscrete result = UnnormalizedDiscrete.Uniform(a.Dimension);
            result.SetToProduct(a, b);
            return result;
        }

        /// <summary>
        /// Sets the parameters to represent the ratio of two unnormalized discrete distributions.
        /// </summary>
        /// <param name="numerator">The first unnormalized discrete distribution</param>
        /// <param name="denominator">The second unnormalized discrete distribution</param>
        /// <param name="forceProper">Ignored</param>
        public void SetToRatio(UnnormalizedDiscrete numerator, UnnormalizedDiscrete denominator, bool forceProper = false)
        {
            if (numerator.IsPointMass)
            {
                if (denominator.IsPointMass)
                {
                    if (numerator.Point == denominator.Point) SetToUniform();
                    else throw new DivideByZeroException();
                }
                else SetTo(numerator);
            }
            else if (denominator.IsPointMass)
            {
                throw new DivideByZeroException();
            }
            else
            {
                logProb.SetToDifference(numerator.logProb, denominator.logProb);
            }
        }

        /// <summary>
        /// Creates an unnormalized discrete distribution which is the ratio of two unnormalized discrete distributions
        /// </summary>
        /// <param name="a">The first distribution</param>
        /// <param name="b">The second distribution</param>
        /// <returns>The resulting unnormalized discrete distribution</returns>
        public static UnnormalizedDiscrete operator /(UnnormalizedDiscrete a, UnnormalizedDiscrete b)
        {
            UnnormalizedDiscrete result = UnnormalizedDiscrete.Uniform(a.Dimension);
            result.SetToRatio(a, b);
            return result;
        }

        public void SetToPower(UnnormalizedDiscrete dist, double exponent)
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
            logProb.Scale(exponent);
        }

        public static UnnormalizedDiscrete operator ^(UnnormalizedDiscrete dist, double exponent)
        {
            UnnormalizedDiscrete result = UnnormalizedDiscrete.Uniform(dist.Dimension);
            result.SetToPower(dist, exponent);
            return result;
        }

        /// <summary>
        /// Override of ToString method
        /// </summary>
        /// <returns>String representation of this instance</returns>
        /// <exclude/>
        public override string ToString()
        {
            return ToString("g4");
        }

        /// <summary>
        /// Override of ToString method which allows custom number formatting
        /// </summary>
        /// <returns>String representation of this instance</returns>
        /// <exclude/>
        public string ToString(string format)
        {
            if (IsPointMass)
            {
                return "UnnormalizedDiscrete.PointMass(" + Point + ")";
            }
            return "UnnormalizedDiscrete(" + GetProbs().ToString(format) + ")";
        }

        /// <summary>
        /// Gets the unnormalized probability vector for this distribution.
        /// </summary>
        /// <returns></returns>
        public DenseVector GetProbs()
        {
            DenseVector p = DenseVector.Zero(logProb.Count);
            p.SetToFunction(logProb, Math.Exp);
            return p;
        }

        /// <summary>
        /// Rescales the unnormalised distribution, so that the max log prob is zero i.e. the max unnormalised prob is 1.
        /// </summary>
        public void SetMaxToZero()
        {
            double d = logProb.Max();
            logProb.SetToFunction(logProb, x => x - d);
        }

        public void SetToSum(double weight1, UnnormalizedDiscrete value1, double weight2, UnnormalizedDiscrete value2)
        {
            Discrete result = Discrete.Uniform(Dimension);
            result.SetToSum(weight1, value1.ToDiscrete(), weight2, value2.ToDiscrete());
            SetLogProbs((DenseVector) result.GetLogProbs());
        }

        public double GetLogAverageOf(UnnormalizedDiscrete that)
        {
            throw new NotImplementedException();
        }

        public double GetAverageLog(UnnormalizedDiscrete that)
        {
            throw new NotImplementedException();
        }

        public int Sample()
        {
            return ToDiscrete().Sample();
        }

        public int Sample(int result)
        {
            return Sample();
        }
    }
}