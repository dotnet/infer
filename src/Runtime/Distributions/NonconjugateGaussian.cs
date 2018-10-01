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
    /// Nonconjugate Gaussian messages for VMP. The mean has a Gaussian distribution and the variance a Gamma distribution. 
    /// </summary>
    [Serializable]
    [DataContract]
    [Quality(QualityBand.Experimental)]
    public struct NonconjugateGaussian : IDistribution<double>,
                                         SettableTo<Gaussian>,
                                         SettableTo<NonconjugateGaussian>,
                                         SettableToRatio<NonconjugateGaussian>,
                                         SettableToPower<NonconjugateGaussian>,
                                         SettableToProduct<NonconjugateGaussian>,
                                         Sampleable<double>,
                                         SettableToWeightedSum<NonconjugateGaussian>,
                                         CanGetLogAverageOf<NonconjugateGaussian>,
                                         CanGetLogAverageOfPower<NonconjugateGaussian>,
                                         CanGetAverageLog<NonconjugateGaussian>,
                                         CanGetMeanAndVarianceOut<double, double>,
                                         Diffable
    {
        /// <summary>
        /// Mean times precision for the mean
        /// </summary>
        [DataMember]
        public double MeanTimesPrecision;

        /// <summary>
        /// Precision for the mean
        /// </summary>
        [DataMember]
        public double Precision;

        /// <summary>
        /// Shape parameter for the variance
        /// </summary>
        [DataMember]
        public double Shape;

        /// <summary>
        /// Rate parameter for the variance
        /// </summary>
        [DataMember]
        public double Rate;

        /// <summary>
        /// Convert to the optimal Gaussian
        /// </summary>
        /// <returns></returns>
        public Gaussian GetGaussian()
        {
            if (Precision == 0.0 || Rate == 0.0)
                return Gaussian.Uniform();
            return Gaussian.FromMeanAndVariance(MeanTimesPrecision/Precision, (Shape - 1)/Rate);
        }

        /// <summary>
        /// Convert to the optimal Gaussian
        /// </summary>
        /// <param name="addEntropy">Whether to include an entropy term</param>
        /// <returns></returns>
        public Gaussian GetGaussian(bool addEntropy)
        {
            if (Precision == 0.0 || Rate == 0.0)
                return Gaussian.Uniform();
            return Gaussian.FromMeanAndVariance(MeanTimesPrecision/Precision, (Shape - 1 + (addEntropy ? .5 : 0))/Rate);
        }

        /// <summary>
        /// Constructs a non-conjugate Gaussian from a Gaussian
        /// </summary>
        /// <param name="that"></param>
        public NonconjugateGaussian(Gaussian that)
            : this()
        {
            SetTo(that);
        }

        /// <summary>
        /// Copy constructor
        /// </summary>
        /// <param name="that"></param>
        public NonconjugateGaussian(NonconjugateGaussian that)
            : this()
        {
            SetTo(that);
        }

        /// <summary>
        /// Constructs a non-conjugate Gaussian distribution its parameters 
        /// </summary>
        /// <param name="meanTimesPrecision">Mean times precision for the mean</param>
        /// <param name="precision">Precision for the mean</param>
        /// <param name="shape">Shape parameter for the variance</param>
        /// <param name="rate">Rate parameter for the variance</param>
        [Construction("MeanTimesPrecision", "Precision", "Shape", "Rate")]
        public NonconjugateGaussian(double meanTimesPrecision, double precision, double shape, double rate)
        {
            MeanTimesPrecision = meanTimesPrecision;
            Precision = precision;
            Shape = shape;
            Rate = rate;
        }

        /// <summary>
        /// Returns true if the distribution is proper
        /// </summary>
        /// <returns></returns>
        public bool IsProper()
        {
            return (Precision > 0 && Shape > 0 && Rate > 0);
        }

        #region ICloneable Members

        object ICloneable.Clone()
        {
            return new NonconjugateGaussian(this);
        }

        #endregion

        #region HasPoint<double> Members

        double HasPoint<double>.Point
        {
            get { throw new NotImplementedException(); }
            set { throw new NotImplementedException(); }
        }

        bool HasPoint<double>.IsPointMass
        {
            get { throw new NotImplementedException(); }
        }

        #endregion

        #region Diffable Members

        /// <summary>
        /// The maximum difference between the parameters of this distribution and that
        /// </summary>
        /// <param name="thatd"></param>
        /// <returns></returns>
        public double MaxDiff(object thatd)
        {
            if (!(thatd is NonconjugateGaussian))
                return Double.PositiveInfinity;

            NonconjugateGaussian that = (NonconjugateGaussian)thatd;

            double diff1 = MMath.AbsDiff(this.MeanTimesPrecision, that.MeanTimesPrecision);
            double diff2 = MMath.AbsDiff(this.Precision, that.Precision);
            double diff3 = MMath.AbsDiff(this.Shape, that.Shape);
            double diff4 = MMath.AbsDiff(this.Rate, that.Rate);
            return Math.Max(Math.Max(diff1, diff2), Math.Max(diff3, diff4));
        }

        #endregion

        #region SettableToUniform Members

        /// <summary>
        /// Sets this to a uniform distribution
        /// </summary>
        public void SetToUniform()
        {
            MeanTimesPrecision = 0;
            Precision = 0;
            Shape = 1;
            Rate = 0;
        }

        /// <summary>
        /// Returns true if distribution is uniform
        /// </summary>
        /// <returns></returns>
        public bool IsUniform()
        {
            return Rate == 0;
        }

        #endregion

        #region CanGetLogProb<double> Members

        /// <summary>
        /// Gets the log probability of the given value
        /// </summary>
        /// <param name="value"></param>
        /// <returns>Not yet implemented</returns>
        double CanGetLogProb<double>.GetLogProb(double value)
        {
            return 0;
        }

        #endregion

        #region SettableTo<Gaussian> Members

        /// <summary>
        /// Sets this non-congugate Gaussian to a Gaussian
        /// </summary>
        /// <param name="value"></param>
        public void SetTo(Gaussian value)
        {
            MeanTimesPrecision = value.MeanTimesPrecision;
            Precision = value.Precision;
            Shape = 1;
            Rate = .5*value.Precision;
        }

        #endregion

        #region SettableTo<NonconjugateGaussian> Members

        /// <summary>
        /// Sets this non-conjugate Gaussian to another non-conjugate Gaussian
        /// </summary>
        /// <param name="value"></param>
        public void SetTo(NonconjugateGaussian value)
        {
            MeanTimesPrecision = value.MeanTimesPrecision;
            Precision = value.Precision;
            Shape = value.Shape;
            Rate = value.Rate;
        }

        #endregion

        #region SettableToProduct<NonconjugateGaussian,NonconjugateGaussian> Members

        /// <summary>
        /// Sets this non-conjugate Gaussian distribution to the product of two others.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        public void SetToProduct(NonconjugateGaussian a, NonconjugateGaussian b)
        {
            MeanTimesPrecision = a.MeanTimesPrecision + b.MeanTimesPrecision;
            Precision = a.Precision + b.Precision;
            Shape = (a.Shape - 1) + (b.Shape - 1) + 1;
            Rate = a.Rate + b.Rate;
        }

        /// <summary>
        /// Product operator
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static NonconjugateGaussian operator *(NonconjugateGaussian a, NonconjugateGaussian b)
        {
            var result = new NonconjugateGaussian();
            result.SetToProduct(a, b);
            return result;
        }

        #endregion

        #region SettableToRatio<NonconjugateGaussian,NonconjugateGaussian> Members

        /// <summary>
        /// Sets this non-conjugate Gaussian distribution to the ratio of two others.
        /// </summary>
        /// <param name="numerator">Numerator</param>
        /// <param name="denominator">Denominator</param>
        /// <param name="forceProper">Ignored</param>
        public void SetToRatio(NonconjugateGaussian numerator, NonconjugateGaussian denominator, bool forceProper = false)
        {
            MeanTimesPrecision = numerator.MeanTimesPrecision - denominator.MeanTimesPrecision;
            Precision = numerator.Precision - denominator.Precision;
            Shape = (numerator.Shape - 1) - (denominator.Shape - 1) + 1;
            Rate = numerator.Rate - denominator.Rate;
        }

        #endregion

        #region SettableToPower<NonconjugateGaussian> Members

        /// <summary>
        /// Sets this non-conjugate Gaussian distribution to the power of another.
        /// </summary>
        /// <param name="value">The </param>
        /// <param name="exponent"></param>
        public void SetToPower(NonconjugateGaussian value, double exponent)
        {
            MeanTimesPrecision = value.MeanTimesPrecision*exponent;
            Precision = value.Precision*exponent;
            Shape = (value.Shape - 1)*exponent + 1;
            Rate = value.Rate*exponent;
        }

        #endregion

        #region Sampleable<double> Members

        /// <summary>
        /// Samples from a non-conjugate Gaussian distribution
        /// </summary>
        /// <returns>Not yet implemented</returns>
        double Sampleable<double>.Sample()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Samples from a non-conjugate Gaussian distribution
        /// </summary>
        /// <param name="result">Where to put the result</param>
        double Sampleable<double>.Sample(double result)
        {
            throw new NotImplementedException();
        }

        #endregion

#pragma warning disable 1591

        #region SettableToWeightedSum<NonconjugateGaussian> Members

        public void SetToSum(double weight1, NonconjugateGaussian value1, double weight2, NonconjugateGaussian value2)
        {
            throw new NotImplementedException();
        }

        #endregion

        #region CanGetLogAverageOf<NonconjugateGaussian> Members

        public double GetLogAverageOf(NonconjugateGaussian that)
        {
            throw new NotImplementedException();
        }

        #endregion

        #region CanGetAverageLog<NonconjugateGaussian> Members

        public double GetAverageLog(NonconjugateGaussian that)
        {
            throw new NotImplementedException();
        }

        #endregion

        /// <summary>
        /// Create a uniform non-conjugate Gaussian distribution
        /// </summary>
        /// <returns></returns>
        [Construction(UseWhen = "IsUniform"), Skip]
        public static NonconjugateGaussian Uniform()
        {
            var result = new NonconjugateGaussian();
            result.SetToUniform();
            return result;
        }

        #region CanGetMeanAndVarianceOut<double,double> Members

        /// <summary>
        /// Gets the mean and variance of this distribution
        /// </summary>
        /// <param name="mean">Output mean</param>
        /// <param name="variance">Output variance</param>
        public void GetMeanAndVariance(out double mean, out double variance)
        {
            mean = MeanTimesPrecision/Precision;
            bool addEntropy = true;
            variance = (Shape - 1 + (addEntropy ? .5 : 0))/Rate;
        }

        #endregion

        /// <summary>
        /// Print details as as string
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            if (IsUniform())
            {
                return "NonconjugateGaussian.Uniform";
            }
            else if (IsProper())
            {
                double m, v;
                GetMeanAndVariance(out m, out v);
                return "NonconjugateGaussian(" + m + ", " + v + ")";
            }
            else
            {
                return "NonconjugateGaussian(" + MeanTimesPrecision + ", " + Precision + ", " + Shape + ", " + Rate + ")";
            }
        }

        public double GetLogAverageOfPower(NonconjugateGaussian that, double power)
        {
            throw new NotImplementedException();
        }
    }
}