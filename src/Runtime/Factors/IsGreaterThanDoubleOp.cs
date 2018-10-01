// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    [FactorMethod(typeof(Factor), "IsGreaterThan", typeof(double), typeof(double))]
    [Quality(QualityBand.Preview)]
    public static class IsGreaterThanDoubleOp
    {
        // Beta ---------------------------------------------------------------------------------------

        public static Bernoulli IsGreaterThanAverageConditional([Proper] Beta a, double b)
        {
            if (a.IsPointMass)
                return Bernoulli.PointMass(a.Point > b);
            return new Bernoulli(1 - a.GetProbLessThan(b));
        }

        public static Bernoulli IsGreaterThanAverageConditional(double a, [Proper] Beta b)
        {
            if (b.IsPointMass)
                return Bernoulli.PointMass(a > b.Point);
            return new Bernoulli(b.GetProbLessThan(a));
        }

        public static Bernoulli IsGreaterThanAverageConditional([Proper] Beta a, [Proper] Beta b)
        {
            if (a.IsPointMass)
                return IsGreaterThanAverageConditional(a.Point, b);
            if (b.IsPointMass)
                return IsGreaterThanAverageConditional(a, b.Point);
            double aVariance = a.GetVariance();
            double bVariance = b.GetVariance();
            int sampleCount = 10000;
            double sum = 0;
            for (int sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
            {
                if (aVariance < bVariance)
                {
                    double aSample = a.Sample();
                    sum += b.GetProbLessThan(aSample);
                }
                else
                {
                    double bSample = b.Sample();
                    sum += 1 - a.GetProbLessThan(bSample);
                }
            }
            double prob = sum / sampleCount;
            return new Bernoulli(prob);
        }

        public static Beta AAverageConditional(bool isGreaterThan, Beta a, double b)
        {
            if ((isGreaterThan && b >= 1) || (!isGreaterThan && b < 0)) throw new AllZeroException();
            if (a.IsPointMass)
            {
                if (isGreaterThan)
                {
                    if (a.Point > b) return Beta.Uniform();
                    else return Beta.PointMass(b);
                }
                else
                {
                    if (a.Point <= b) return Beta.Uniform();
                    else return Beta.PointMass(b);
                }
            }
            else if(a.IsUniform())
            {
                if(isGreaterThan)
                {
                    return Beta.FromMeanAndVariance((b + 1) / 2, (1 - b) * (1 - b) / 12);
                }
                else
                {
                    return Beta.FromMeanAndVariance(b / 2, b * b / 12);
                }
            }
            throw new NotImplementedException();
        }

        public static Beta AAverageConditional([SkipIfUniform] Bernoulli isGreaterThan, Beta a, Beta b)
        {
            throw new NotImplementedException();
        }

        public static Beta BAverageConditional(bool isGreaterThan, double a, Beta b)
        {
            if ((isGreaterThan && a <= 0) || (!isGreaterThan && a > 1)) throw new AllZeroException();
            if (b.IsPointMass)
            {
                if (isGreaterThan)
                {
                    if (a > b.Point) return Beta.Uniform();
                    else return Beta.PointMass(a);
                }
                else
                {
                    if (a <= b.Point) return Beta.Uniform();
                    else return Beta.PointMass(a);
                }
            }
            else if(b.IsUniform())
            {
                if (isGreaterThan)
                {
                    return Beta.FromMeanAndVariance(a / 2, a * a / 12);
                }
                else
                {
                    return Beta.FromMeanAndVariance((a + 1) / 2, (1 - a) * (1 - a) / 12);
                }
            }
            throw new NotImplementedException();
        }

        public static Beta BAverageConditional([SkipIfUniform] Bernoulli isGreaterThan, Beta a, Beta b)
        {
            throw new NotImplementedException();
        }

        // Gamma ---------------------------------------------------------------------------------------

        public static Bernoulli IsGreaterThanAverageConditional([Proper] Gamma a, double b)
        {
            if (a.IsPointMass)
                return Bernoulli.PointMass(a.Point > b);
            return new Bernoulli(1 - a.GetProbLessThan(b));
        }

        public static Bernoulli IsGreaterThanAverageConditional(double a, [Proper] Gamma b)
        {
            if (b.IsPointMass)
                return Bernoulli.PointMass(a > b.Point);
            return new Bernoulli(b.GetProbLessThan(a));
        }

        public static Bernoulli IsGreaterThanAverageConditional([Proper] Gamma a, [Proper] Gamma b)
        {
            if (a.IsPointMass)
                return IsGreaterThanAverageConditional(a.Point, b);
            if (b.IsPointMass)
                return IsGreaterThanAverageConditional(a, b.Point);
            if (a.Rate == b.Rate)
            {
                // If a and b have the same Rate, then c = b/(b+a) is Beta(b.Shape, a.Shape).
                // (a > b) is equivalent to (c < 0.5).
                Beta c = new Beta(b.Shape, a.Shape);
                return new Bernoulli(c.GetProbLessThan(0.5));
            }
            double aVariance = a.GetVariance();
            double bVariance = b.GetVariance();
            int sampleCount = 10000;
            double sum = 0;
            for (int sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
            {
                if (aVariance < bVariance)
                {
                    double aSample = a.Sample();
                    sum += b.GetProbLessThan(aSample);
                }
                else
                {
                    double bSample = b.Sample();
                    sum += 1 - a.GetProbLessThan(bSample);
                }
            }
            double prob = sum / sampleCount;
            return new Bernoulli(prob);
        }

        public static Gamma AAverageConditional(bool isGreaterThan, [Proper] Gamma a, double b)
        {
            if ((isGreaterThan && b > double.MaxValue) || (!isGreaterThan && b < 0)) throw new AllZeroException();
            if (a.IsPointMass)
            {
                if (isGreaterThan)
                {
                    if (a.Point > b) return Gamma.Uniform();
                    else return Gamma.PointMass(b);
                }
                else
                {
                    if (a.Point <= b) return Gamma.Uniform();
                    else return Gamma.PointMass(b);
                }
            }
            throw new NotImplementedException();
        }

        public static Gamma AAverageConditional([SkipIfUniform] Bernoulli isGreaterThan, Gamma a, Gamma b)
        {
            throw new NotImplementedException();
        }

        public static Gamma BAverageConditional(bool isGreaterThan, double a, [Proper] Gamma b)
        {
            if (isGreaterThan && a <= 0) throw new AllZeroException();
            if (b.IsPointMass)
            {
                if (isGreaterThan)
                {
                    if (a > b.Point) return Gamma.Uniform();
                    else return Gamma.PointMass(a);
                }
                else
                {
                    if (a <= b.Point) return Gamma.Uniform();
                    else return Gamma.PointMass(a);
                }
            }
            throw new NotImplementedException();
        }

        public static Gamma BAverageConditional([SkipIfUniform] Bernoulli isGreaterThan, Gamma a, Gamma b)
        {
            throw new NotImplementedException();
        }

        public static TruncatedGamma AAverageConditional(bool isGreaterThan, double b)
        {
            if (isGreaterThan) return new TruncatedGamma(Gamma.Uniform(), b, double.PositiveInfinity);
            else return new TruncatedGamma(Gamma.Uniform(), 0, b);
        }

        public static TruncatedGamma BAverageConditional(bool isGreaterThan, double a)
        {
            return AAverageConditional(!isGreaterThan, a);
        }
    }
}
