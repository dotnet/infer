// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/doc/*'/>
    /// <remarks>A and B need not have the same dimension.</remarks>
    [FactorMethod(typeof(Factor), "IsGreaterThan", typeof(int), typeof(int))]
    [Quality(QualityBand.Preview)]
    public static class IsGreaterThanOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="LogAverageFactor(bool, int, int)"]/*'/>
        public static double LogAverageFactor(bool isGreaterThan, int a, int b)
        {
            return (isGreaterThan == Factor.IsGreaterThan(a, b)) ? 0.0 : Double.NegativeInfinity;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="LogEvidenceRatio(bool, int, int)"]/*'/>
        public static double LogEvidenceRatio(bool isGreaterThan, int a, int b)
        {
            return LogAverageFactor(isGreaterThan, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="AverageLogFactor(bool, int, int)"]/*'/>
        public static double AverageLogFactor(bool isGreaterThan, int a, int b)
        {
            return LogAverageFactor(isGreaterThan, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="LogAverageFactor(Bernoulli, int, int)"]/*'/>
        public static double LogAverageFactor(Bernoulli isGreaterThan, int a, int b)
        {
            return isGreaterThan.GetLogProb(Factor.IsGreaterThan(a, b));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="LogAverageFactor(Bernoulli, Discrete, Bernoulli)"]/*'/>
        public static double LogAverageFactor(Bernoulli isGreaterThan, Discrete a, [Fresh] Bernoulli to_isGreaterThan)
        {
            return to_isGreaterThan.GetLogAverageOf(isGreaterThan);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="LogAverageFactor(Bernoulli, int, Discrete, Bernoulli)"]/*'/>
        public static double LogAverageFactor(Bernoulli isGreaterThan, int a, Discrete b, [Fresh] Bernoulli to_isGreaterThan)
        {
            return to_isGreaterThan.GetLogAverageOf(isGreaterThan);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="LogEvidenceRatio(Bernoulli)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Bernoulli isGreaterThan)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="LogAverageFactor(bool, Discrete, Discrete)"]/*'/>
        public static double LogAverageFactor(bool isGreaterThan, Discrete a, Discrete b)
        {
            if (isGreaterThan)
            {
                // a > b
                double sum = 0.0;
                for (int i = 1; i < a.Dimension; i++)
                {
                    double bSum = 0.0;
                    for (int j = 0; (j < b.Dimension) && (j < i); j++)
                    {
                        bSum += b[j];
                    }
                    sum += a[i] * bSum;
                }
                return Math.Log(sum);
            }
            else
            {
                // a <= b
                double sum = 0.0;
                for (int i = 0; i < a.Dimension; i++)
                {
                    double bSum = 0.0;
                    for (int j = i; j < b.Dimension; j++)
                    {
                        bSum += b[j];
                    }
                    sum += a[i] * bSum;
                }
                return Math.Log(sum);
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="LogAverageFactor(bool, int, Discrete)"]/*'/>
        public static double LogAverageFactor(bool isGreaterThan, int a, Discrete b)
        {
            if (isGreaterThan)
            {
                // a > b
                double sum = 0.0;
                for (int j = 0; (j < b.Dimension) && (j < a); j++)
                {
                    sum += b[j];
                }
                return Math.Log(sum);
            }
            else
            {
                // a <= b
                double sum = 0.0;
                for (int j = a; j < b.Dimension; j++)
                {
                    sum += b[j];
                }
                return Math.Log(sum);
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="LogAverageFactor(bool, Discrete, int)"]/*'/>
        public static double LogAverageFactor(bool isGreaterThan, Discrete a, int b)
        {
            if (isGreaterThan)
            {
                // a > b
                double sum = 0.0;
                for (int i = b + 1; i < a.Dimension; i++)
                {
                    sum += a[i];
                }
                return Math.Log(sum);
            }
            else
            {
                // a <= b
                double sum = 0.0;
                for (int i = 0; (i < a.Dimension) && (i <= b); i++)
                {
                    sum += a[i];
                }
                return Math.Log(sum);
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="LogEvidenceRatio(bool, Discrete, Discrete)"]/*'/>
        public static double LogEvidenceRatio(bool isGreaterThan, Discrete a, Discrete b)
        {
            return LogAverageFactor(isGreaterThan, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="LogEvidenceRatio(bool, int, Discrete)"]/*'/>
        public static double LogEvidenceRatio(bool isGreaterThan, int a, Discrete b)
        {
            return LogAverageFactor(isGreaterThan, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="LogEvidenceRatio(bool, Discrete, int)"]/*'/>
        public static double LogEvidenceRatio(bool isGreaterThan, Discrete a, int b)
        {
            return LogAverageFactor(isGreaterThan, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="IsGreaterThanAverageConditional(int, Discrete)"]/*'/>
        public static Bernoulli IsGreaterThanAverageConditional(int a, Discrete b)
        {
            if (b.IsPointMass)
                return Bernoulli.PointMass(a > b.Point);
            double sum = 0.0;
            for (int i = 0; (i < a) && (i < b.Dimension); i++)
            {
                sum += b[i];
            }
            if (sum > 1)
                sum = 1; // this can happen due to round-off errors
            return new Bernoulli(sum);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="IsGreaterThanAverageConditional(Discrete, int)"]/*'/>
        public static Bernoulli IsGreaterThanAverageConditional(Discrete a, int b)
        {
            if (a.IsPointMass)
                return Bernoulli.PointMass(a.Point > b);
            double sum = 0.0;
            for (int i = b + 1; i < a.Dimension; i++)
            {
                sum += a[i];
            }
            if (sum > 1)
                sum = 1; // this can happen due to round-off errors
            return new Bernoulli(sum);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="IsGreaterThanAverageConditional(Discrete, Discrete)"]/*'/>
        public static Bernoulli IsGreaterThanAverageConditional(Discrete a, Discrete b)
        {
            if (a.IsPointMass)
                return IsGreaterThanAverageConditional(a.Point, b);
            if (b.IsPointMass)
                return IsGreaterThanAverageConditional(a, b.Point);
            double sum = 0.0;
            for (int i = 1; i < a.Dimension; i++)
            {
                double xsum = 0.0;
                for (int j = 0; (j < i) && (j < b.Dimension); j++)
                {
                    xsum += b[j];
                }
                sum += xsum * a[i];
            }
            if (sum > 1)
                sum = 1; // this can happen due to round-off errors
            return new Bernoulli(sum);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="AAverageConditional(Bernoulli, Discrete, Discrete)"]/*'/>
        public static Discrete AAverageConditional([SkipIfUniform] Bernoulli isGreaterThan, Discrete b, Discrete result)
        {
            if (b.IsPointMass)
                return AAverageConditional(isGreaterThan, b.Point, result);
            Vector aProbs = result.GetWorkspace();
            double probTrue = isGreaterThan.GetProbTrue();
            double probFalse = 1 - probTrue;
            for (int i = 0; i < aProbs.Count; i++)
            {
                double sum1 = 0.0;
                int j = 0;
                for (; (j < i) && (j < b.Dimension); j++)
                {
                    sum1 += b[j];
                }
                double sum0 = 0.0;
                for (; j < b.Dimension; j++)
                {
                    sum0 += b[j];
                }
                aProbs[i] = probTrue * sum1 + probFalse * sum0;
            }
            result.SetProbs(aProbs);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="AAverageConditional(Bernoulli, int, Discrete)"]/*'/>
        public static Discrete AAverageConditional([SkipIfUniform] Bernoulli isGreaterThan, int b, Discrete result)
        {
            double probTrue = isGreaterThan.GetProbTrue();
            double probFalse = 1 - probTrue;
            Vector aProbs = result.GetWorkspace();
            for (int i = 0; i < aProbs.Count; i++)
            {
                aProbs[i] = (i > b) ? probTrue : probFalse;
            }
            result.SetProbs(aProbs);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="AAverageConditional(bool, Discrete, Discrete)"]/*'/>
        public static Discrete AAverageConditional(bool isGreaterThan, Discrete b, Discrete result)
        {
            return AAverageConditional(Bernoulli.PointMass(isGreaterThan), b, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="AAverageConditional(bool, int, Discrete)"]/*'/>
        public static Discrete AAverageConditional(bool isGreaterThan, int b, Discrete result)
        {
            return AAverageConditional(Bernoulli.PointMass(isGreaterThan), b, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="BAverageConditional(Bernoulli, Discrete, Discrete)"]/*'/>
        public static Discrete BAverageConditional([SkipIfUniform] Bernoulli isGreaterThan, Discrete a, Discrete result)
        {
            if (a.IsPointMass)
                return BAverageConditional(isGreaterThan, a.Point, result);
            Vector bProbs = result.GetWorkspace();
            double probTrue = isGreaterThan.GetProbTrue();
            double probFalse = 1 - probTrue;
            for (int j = 0; j < bProbs.Count; j++)
            {
                double sum0 = 0.0;
                int i = 0;
                for (; (i <= j) && (i < a.Dimension); i++)
                {
                    sum0 += a[i];
                }
                double sum1 = 0.0;
                for (; i < a.Dimension; i++)
                {
                    sum1 += a[i];
                }
                bProbs[j] = probTrue * sum1 + probFalse * sum0;
            }
            result.SetProbs(bProbs);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="BAverageConditional(Bernoulli, int, Discrete)"]/*'/>
        public static Discrete BAverageConditional([SkipIfUniform] Bernoulli isGreaterThan, int a, Discrete result)
        {
            Vector bProbs = result.GetWorkspace();
            double probTrue = isGreaterThan.GetProbTrue();
            double probFalse = 1 - probTrue;
            for (int j = 0; j < bProbs.Count; j++)
            {
                bProbs[j] = (a > j) ? probTrue : probFalse;
            }
            result.SetProbs(bProbs);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="BAverageConditional(bool, Discrete, Discrete)"]/*'/>
        public static Discrete BAverageConditional(bool isGreaterThan, Discrete a, Discrete result)
        {
            return BAverageConditional(Bernoulli.PointMass(isGreaterThan), a, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="BAverageConditional(bool, int, Discrete)"]/*'/>
        public static Discrete BAverageConditional(bool isGreaterThan, int a, Discrete result)
        {
            return BAverageConditional(Bernoulli.PointMass(isGreaterThan), a, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="IsGreaterThanAverageConditional(Binomial, int)"]/*'/>
        public static Bernoulli IsGreaterThanAverageConditional([Proper] Binomial a, int b)
        {
            if (a.IsPointMass)
                return Bernoulli.PointMass(a.Point > b);
            if (a.A == 1 && a.B == 1)
            {
                if (b < 0)
                    return Bernoulli.PointMass(true);
                else if (b >= a.TrialCount)
                    return Bernoulli.PointMass(false);
                else
                    return new Bernoulli(MMath.Beta(a.ProbSuccess, b + 1, a.TrialCount - b));
            }
            else
            {
                double sum = 0;
                for (int i = 0; i <= b; i++)
                {
                    sum += Math.Exp(a.GetLogProb(i));
                }
                if (sum > 1)
                    sum = 1; // this can happen due to round-off errors
                return new Bernoulli(1 - sum);
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="IsGreaterThanAverageConditional(int, Binomial)"]/*'/>
        public static Bernoulli IsGreaterThanAverageConditional(int a, [Proper] Binomial b)
        {
            if (b.IsPointMass)
                return Bernoulli.PointMass(a > b.Point);
            if (b.A == 1 && b.B == 1)
            {
                if (a <= 0)
                    return Bernoulli.PointMass(false);
                else if (a > b.TrialCount)
                    return Bernoulli.PointMass(true);
                else
                    return new Bernoulli(MMath.Beta(1 - b.ProbSuccess, b.TrialCount - a + 1, a));
            }
            else
            {
                double sum = 0;
                for (int i = 0; i < a; i++)
                {
                    sum += Math.Exp(b.GetLogProb(i));
                }
                if (sum > 1)
                    sum = 1; // this can happen due to round-off errors
                return new Bernoulli(sum);
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="IsGreaterThanAverageConditional(Poisson, int)"]/*'/>
        public static Bernoulli IsGreaterThanAverageConditional([Proper] Poisson a, int b)
        {
            if (a.IsPointMass)
                return Bernoulli.PointMass(a.Point > b);
            if (a.Precision == 1)
            {
                if (b < 0)
                    return Bernoulli.PointMass(true);
                else
                    return new Bernoulli(MMath.GammaLower(b + 1, a.Rate));
            }
            else
            {
                double sum = 0;
                for (int i = 0; i <= b; i++)
                {
                    sum += Math.Exp(a.GetLogProb(i));
                }
                if (sum > 1)
                    sum = 1; // this can happen due to round-off errors
                return new Bernoulli(1 - sum);
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="IsGreaterThanAverageConditional(int, Poisson)"]/*'/>
        public static Bernoulli IsGreaterThanAverageConditional(int a, [Proper] Poisson b)
        {
            if (b.IsPointMass)
                return Bernoulli.PointMass(a > b.Point);
            if (b.Precision == 1)
            {
                if (a <= 0)
                    return Bernoulli.PointMass(false);
                else
                    return new Bernoulli(MMath.GammaUpper(a, b.Rate));
            }
            else
            {
                double sum = 0;
                for (int i = 0; i < a; i++)
                {
                    sum += Math.Exp(b.GetLogProb(i));
                }
                if (sum > 1)
                    sum = 1; // this can happen due to round-off errors
                return new Bernoulli(sum);
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="AAverageConditional(Bernoulli, Poisson, int)"]/*'/>
        public static Poisson AAverageConditional([SkipIfUniform] Bernoulli isGreaterThan, [Proper] Poisson a, int b)
        {
            if (a.IsPointMass || !a.IsProper())
                return Poisson.Uniform();
            double aMean = a.GetMean();
            double sum = 0;
            double asum = 0;
            if (a.Precision == 1)
            {
                if (b >= 0)
                {
                    sum = MMath.GammaUpper(b + 1, a.Rate);
                    asum = (sum - Math.Exp(a.GetLogProb(b))) * a.Rate;
                }
            }
            else
            {
                for (int i = 0; i <= b; i++)
                {
                    double p = Math.Exp(a.GetLogProb(i));
                    sum += p;
                    asum += i * p;
                }
                if (sum > 1)
                    sum = 1; // this can happen due to round-off errors
                if (asum > aMean)
                    asum = aMean;
            }
            double pT = isGreaterThan.GetProbTrue();
            double pF = 1 - pT;
            double Z = pF * sum + pT * (1 - sum);
            double aZ = pF * asum + pT * (aMean - asum);
            Poisson result = new Poisson(aZ / Z);
            result.SetToRatio(result, a, false);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="AAverageConditional(Bernoulli, Poisson, int)"]/*'/>
        public static Poisson BAverageConditional([SkipIfUniform] Bernoulli isGreaterThan, int a, [Proper] Poisson b)
        {
            if (b.IsPointMass || !b.IsProper())
                return Poisson.Uniform();
            double bMean = b.GetMean();
            double sum = 0;
            double bsum = 0;
            if (b.Precision == 1)
            {
                if (a > 0)
                {
                    sum = MMath.GammaUpper(a, b.Rate);
                    bsum = (sum - Math.Exp(b.GetLogProb(a - 1))) * b.Rate;
                }
            }
            else
            {
                for (int i = 0; i < a; i++)
                {
                    double p = Math.Exp(b.GetLogProb(i));
                    sum += p;
                    bsum += i * p;
                }
                if (sum > 1)
                    sum = 1; // this can happen due to round-off errors
                if (bsum > bMean)
                    bsum = bMean;
            }
            double pT = isGreaterThan.GetProbTrue();
            double pF = 1 - pT;
            double Z = pT * sum + pF * (1 - sum);
            double bZ = pT * bsum + pF * (bMean - bsum);
            Poisson result = new Poisson(bZ / Z);
            result.SetToRatio(result, b, false);
            return result;
        }

        // VMP /////////////////////////////////////////////////////////////////////////////////////////////////////////

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="AverageLogFactor(Bernoulli, Discrete, Discrete)"]/*'/>
        [Skip]
        public static double AverageLogFactor(Bernoulli isGreaterThan, Discrete a, Discrete b)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="AverageLogFactor(Bernoulli, int, Discrete)"]/*'/>
        [Skip]
        public static double AverageLogFactor(Bernoulli isGreaterThan, int a, Discrete b)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="AverageLogFactor(Bernoulli, Discrete, int)"]/*'/>
        [Skip]
        public static double AverageLogFactor(Bernoulli isGreaterThan, Discrete a, int b)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="AverageLogFactor(Bernoulli, int, int)"]/*'/>
        [Skip]
        public static double AverageLogFactor(Bernoulli isGreaterThan, int a, int b)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="IsGreaterThanAverageLogarithm(int, Discrete)"]/*'/>
        public static Bernoulli IsGreaterThanAverageLogarithm(int a, Discrete b)
        {
            return IsGreaterThanAverageConditional(a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="IsGreaterThanAverageLogarithm(Discrete, int)"]/*'/>
        public static Bernoulli IsGreaterThanAverageLogarithm(Discrete a, int b)
        {
            return IsGreaterThanAverageConditional(a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="IsGreaterThanAverageLogarithm(Discrete, Discrete)"]/*'/>
        public static Bernoulli IsGreaterThanAverageLogarithm(Discrete a, Discrete b)
        {
            return IsGreaterThanAverageConditional(a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="AAverageLogarithm(Bernoulli, Discrete, Discrete)"]/*'/>
        public static Discrete AAverageLogarithm([SkipIfUniform] Bernoulli isGreaterThan, Discrete b, Discrete result)
        {
            if (b.IsPointMass)
                return AAverageLogarithm(isGreaterThan, b.Point, result);
            if (isGreaterThan.IsPointMass)
                return AAverageLogarithm(isGreaterThan.Point, b, result);
            // f(a,b) = p(c=1) I(a > b) + p(c=0) I(a <= b)
            // message to a = exp(sum_b q(b) log f(a,b))
            Vector aProbs = result.GetWorkspace();
            double logProbTrue = isGreaterThan.GetLogProbTrue();
            double logProbFalse = isGreaterThan.GetLogProbFalse();
            for (int i = 0; i < aProbs.Count; i++)
            {
                double sum = 0.0;
                int j = 0;
                for (; (j < i) && (j < b.Dimension); j++)
                {
                    sum += logProbTrue * b[j];
                }
                for (; j < b.Dimension; j++)
                {
                    sum += logProbFalse * b[j];
                }
                aProbs[i] = Math.Exp(sum);
            }
            result.SetProbs(aProbs);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="BAverageLogarithm(Bernoulli, Discrete, Discrete)"]/*'/>
        public static Discrete BAverageLogarithm([SkipIfUniform] Bernoulli isGreaterThan, Discrete a, Discrete result)
        {
            if (a.IsPointMass)
                return BAverageLogarithm(isGreaterThan, a.Point, result);
            if (isGreaterThan.IsPointMass)
                return BAverageLogarithm(isGreaterThan.Point, a, result);
            // f(a,b) = p(c=1) I(a > b) + p(c=0) I(a <= b)
            // message to b = exp(sum_a q(a) log f(a,b))
            Vector bProbs = result.GetWorkspace();
            double logProbTrue = isGreaterThan.GetLogProbTrue();
            double logProbFalse = isGreaterThan.GetLogProbFalse();
            for (int j = 0; j < bProbs.Count; j++)
            {
                double sum = 0.0;
                int i = 0;
                for (; (i <= j) && (i < a.Dimension); i++)
                {
                    sum += logProbFalse * a[i];
                }
                for (; i < a.Dimension; i++)
                {
                    sum += logProbTrue * a[i];
                }
                bProbs[j] = Math.Exp(sum);
            }
            result.SetProbs(bProbs);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="AAverageLogarithm(Bernoulli, int, Discrete)"]/*'/>
        public static Discrete AAverageLogarithm([SkipIfUniform] Bernoulli isGreaterThan, int b, Discrete result)
        {
            return AAverageConditional(isGreaterThan, b, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="BAverageLogarithm(Bernoulli, int, Discrete)"]/*'/>
        public static Discrete BAverageLogarithm([SkipIfUniform] Bernoulli isGreaterThan, int a, Discrete result)
        {
            return BAverageConditional(isGreaterThan, a, result);
        }

        private const string NotSupportedMessage = "Variational Message Passing does not support an IsGreaterThan factor with fixed output.";

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="AverageLogFactor(bool, Discrete, Discrete)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static double AverageLogFactor(bool isGreaterThan, Discrete a, Discrete b)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="AverageLogFactor(bool, int, Discrete)"]/*'/>
        public static double AverageLogFactor(bool isGreaterThan, int a, Discrete b)
        {
            return LogAverageFactor(isGreaterThan, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="AverageLogFactor(bool, Discrete, int)"]/*'/>
        public static double AverageLogFactor(bool isGreaterThan, Discrete a, int b)
        {
            return LogAverageFactor(isGreaterThan, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="AAverageLogarithm(bool, Discrete, Discrete)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Discrete AAverageLogarithm(bool isGreaterThan, Discrete b, Discrete result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="BAverageLogarithm(bool, Discrete, Discrete)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Discrete BAverageLogarithm(bool isGreaterThan, Discrete a, Discrete result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="AAverageLogarithm(bool, int, Discrete)"]/*'/>
        public static Discrete AAverageLogarithm(bool isGreaterThan, int b, Discrete result)
        {
            return AAverageConditional(isGreaterThan, b, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsGreaterThanOp"]/message_doc[@name="BAverageLogarithm(bool, int, Discrete)"]/*'/>
        public static Discrete BAverageLogarithm(bool isGreaterThan, int a, Discrete result)
        {
            return BAverageConditional(isGreaterThan, a, result);
        }
    }
}
