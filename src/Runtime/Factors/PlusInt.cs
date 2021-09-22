// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Plus", typeof(int), typeof(int))]
    [FactorMethod(new[] { "A", "Sum", "B" }, typeof(Factor), "Difference", typeof(int), typeof(int))]
    [Quality(QualityBand.Stable)]
    public static class IntegerPlusOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/message_doc[@name="LogAverageFactor(Discrete, int, int)"]/*'/>
        public static double LogAverageFactor(Discrete sum, int a, int b)
        {
            return sum.GetLogProb(Factor.Plus(a, b));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/message_doc[@name="LogAverageFactor(Discrete, Discrete, Discrete)"]/*'/>
        public static double LogAverageFactor(Discrete sum, Discrete a, [Fresh] Discrete to_sum)
        {
            return to_sum.GetLogAverageOf(sum);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/message_doc[@name="LogAverageFactor(Discrete, int, Discrete, Discrete)"]/*'/>
        public static double LogAverageFactor(Discrete sum, int a, Discrete b, [Fresh] Discrete to_sum)
        {
            return to_sum.GetLogAverageOf(sum);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/message_doc[@name="LogEvidenceRatio(Discrete)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Discrete sum)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/message_doc[@name="LogAverageFactor(int, Discrete, Discrete)"]/*'/>
        public static double LogAverageFactor(int sum, Discrete a, Discrete b)
        {
            if (a.IsPointMass)
                return LogAverageFactor(sum, a.Point, b);
            double z = 0.0;
            for (int i = 0; (i < a.Dimension) && (sum - i < b.Dimension); i++)
            {
                z += a[i] * b[sum - i];
            }
            return Math.Log(z);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/message_doc[@name="LogAverageFactor(int, int, Discrete)"]/*'/>
        public static double LogAverageFactor(int sum, int a, Discrete b)
        {
            if (b.IsPointMass)
                return LogAverageFactor(sum, a, b.Point);
            int j = sum - a;
            if (j < b.Dimension)
                return b.GetLogProb(j);
            else
                return Double.NegativeInfinity;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/message_doc[@name="LogAverageFactor(int, Discrete, int)"]/*'/>
        public static double LogAverageFactor(int sum, Discrete a, int b)
        {
            if (a.IsPointMass)
                return LogAverageFactor(sum, a.Point, b);
            int i = sum - b;
            if (i < a.Dimension)
                return a.GetLogProb(i);
            else
                return Double.NegativeInfinity;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/message_doc[@name="LogAverageFactor(int, int, int)"]/*'/>
        public static double LogAverageFactor(int sum, int a, int b)
        {
            return (sum == a + b) ? 0.0 : Double.NegativeInfinity;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/message_doc[@name="LogEvidenceRatio(int, Discrete, Discrete)"]/*'/>
        public static double LogEvidenceRatio(int sum, Discrete a, Discrete b)
        {
            return LogAverageFactor(sum, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/message_doc[@name="LogEvidenceRatio(int, int, Discrete)"]/*'/>
        public static double LogEvidenceRatio(int sum, int a, Discrete b)
        {
            return LogAverageFactor(sum, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/message_doc[@name="LogEvidenceRatio(int, Discrete, int)"]/*'/>
        public static double LogEvidenceRatio(int sum, Discrete a, int b)
        {
            return LogAverageFactor(sum, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/message_doc[@name="LogEvidenceRatio(int, int, int)"]/*'/>
        public static double LogEvidenceRatio(int sum, int a, int b)
        {
            return LogAverageFactor(sum, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/message_doc[@name="SumAverageConditional(Discrete, Discrete, Discrete)"]/*'/>
        public static Discrete SumAverageConditional(Discrete a, Discrete b, Discrete result)
        {
            if (a.IsPointMass)
                return SumAverageConditional(a.Point, b, result);
            if (b.IsPointMass)
                return SumAverageConditional(a, b.Point, result);
            Vector probs = result.GetWorkspace();
            // Iterating in reverse order ensures that a[i] and b[i] are never read after probs[i] is written, so they can share the same array.
            for (int sum = probs.Count - 1; sum >= 0; sum--)
            {
                double p = 0;
                for (int i = Math.Max(0, sum - b.Dimension + 1); i < a.Dimension; i++)
                {
                    if (i > sum) break;
                    int j = sum - i;
                    // j >= 0 implies sum >= i
                    // j < b.Dimension implies sum < i + b.Dimension
                    p += a[i] * b[j];
                }
                probs[sum] = p;
            }
            result.SetProbs(probs);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/message_doc[@name="SumAverageConditional(int, Discrete, Discrete)"]/*'/>
        public static Discrete SumAverageConditional(int a, Discrete b, Discrete result)
        {
            Vector probs = result.GetWorkspace();
            int minSum = Math.Max(0, a);
            int maxSumPlus1 = Math.Min(probs.Count, b.Dimension + a);
            bool sameObject = ReferenceEquals(b, result);
            if (!sameObject)
            {
                probs.SetAllElementsTo(0.0);
            }
            if (a <= 0)
            {
                for (int sum = minSum; sum < maxSumPlus1; sum++)
                {
                    probs[sum] = b[sum - a];
                }
            }
            else
            {
                // Iterating in reverse order ensures that b[i] is never read after probs[i] is written, so they can share the same array.
                for (int sum = maxSumPlus1 - 1; sum >= minSum; sum--)
                {
                    probs[sum] = b[sum - a];
                }
            }
            if (sameObject)
            {
                for (int sum = 0; sum < minSum; sum++)
                {
                    probs[sum] = 0;
                }
                for (int sum = maxSumPlus1; sum < probs.Count; sum++)
                {
                    probs[sum] = 0;
                }
            }
            result.SetProbs(probs);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/message_doc[@name="SumAverageConditional(Discrete, int, Discrete)"]/*'/>
        public static Discrete SumAverageConditional(Discrete a, int b, Discrete result)
        {
            return SumAverageConditional(b, a, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/message_doc[@name="SumAverageConditionalInit(Discrete, Discrete)"]/*'/>
        [Skip]
        public static Discrete SumAverageConditionalInit([IgnoreDependency] Discrete a, [IgnoreDependency] Discrete b)
        {
            return Discrete.Uniform(a.Dimension + b.Dimension - 1);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/message_doc[@name="SumAverageConditionalInit(Discrete, int)"]/*'/>
        [Skip]
        public static Discrete SumAverageConditionalInit([IgnoreDependency] Discrete a, [IgnoreDependency] int b)
        {
            return Discrete.Uniform(a.Dimension + b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/message_doc[@name="SumAverageConditionalInit(int, Discrete)"]/*'/>
        [Skip]
        public static Discrete SumAverageConditionalInit([IgnoreDependency] int a, [IgnoreDependency] Discrete b)
        {
            return Discrete.Uniform(a + b.Dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/message_doc[@name="AAverageConditional(Discrete, Discrete, Discrete)"]/*'/>
        public static Discrete AAverageConditional(Discrete sum, Discrete b, Discrete result)
        {
            if (ReferenceEquals(sum, result)) throw new ArgumentException("result and sum are the same object", nameof(result));
            if (ReferenceEquals(b, result)) throw new ArgumentException("result and b are the same object", nameof(result));
            // message to a = sum_b p(sum = a+b) p(b)
            Vector probs = result.GetWorkspace();
            for (int i = 0; i < result.Dimension; i++)
            {
                double p = 0.0;
                int maxPlus1 = sum.Dimension - i;
                for (int j = 0; (j < b.Dimension) && (j < maxPlus1); j++)
                {
                    p += b[j] * sum[i + j];
                }
                probs[i] = p;
            }
            result.SetProbs(probs);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/message_doc[@name="AAverageConditional(Discrete, int, Discrete)"]/*'/>
        public static Discrete AAverageConditional(Discrete sum, int b, Discrete result)
        {
            return SumAverageConditional(sum, -b, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/message_doc[@name="AAverageConditional(int, Discrete, Discrete)"]/*'/>
        public static Discrete AAverageConditional(int sum, Discrete b, Discrete result)
        {
            if (ReferenceEquals(b, result)) throw new ArgumentException("result and b are the same object", nameof(result));
            // message to a = p(b = sum-a)
            Vector probs = result.GetWorkspace();
            int min = sum - (b.Dimension - 1);
            min = Math.Max(min, 0); // The domain of Discrete doesn't include negative numbers
            for (int i = 0; i < min; i++)
            {
                probs[i] = 0.0;
            }
            for (int i = min; (i < result.Dimension) && (i <= sum); i++)
            {
                probs[i] = b[sum - i];
            }
            result.SetProbs(probs);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/message_doc[@name="AAverageConditional(int, int, Discrete)"]/*'/>
        public static Discrete AAverageConditional(int sum, int b, Discrete result)
        {
            int a = sum - b;
            if (a < 0 || a > result.Dimension)
                throw new AllZeroException();
            result.Point = a;
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/message_doc[@name="BAverageConditional(Discrete, Discrete, Discrete)"]/*'/>
        public static Discrete BAverageConditional(Discrete sum, Discrete a, Discrete result)
        {
            return AAverageConditional(sum, a, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/message_doc[@name="BAverageConditional(Discrete, int, Discrete)"]/*'/>
        public static Discrete BAverageConditional(Discrete sum, int a, Discrete result)
        {
            return AAverageConditional(sum, a, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/message_doc[@name="BAverageConditional(int, Discrete, Discrete)"]/*'/>
        public static Discrete BAverageConditional(int sum, Discrete a, Discrete result)
        {
            return AAverageConditional(sum, a, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/message_doc[@name="BAverageConditional(int, int, Discrete)"]/*'/>
        public static Discrete BAverageConditional(int sum, int a, Discrete result)
        {
            return AAverageConditional(sum, a, result);
        }

        // Poisson case -----------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/message_doc[@name="SumAverageConditional(Poisson, Poisson)"]/*'/>
        public static Poisson SumAverageConditional(Poisson a, Poisson b)
        {
            if (a.Precision != 1 || b.Precision != 1)
                throw new NotImplementedException("precision != 1 not implemented");
            return new Poisson(a.Rate + b.Rate);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/message_doc[@name="AAverageConditional(int, Poisson, Poisson)"]/*'/>
        public static Poisson AAverageConditional(int sum, Poisson a, Poisson b)
        {
            // a has rate r
            // b has rate t
            // p(a) =propto r^a/a! t^(s-a)/(s-a)! =propto Binom(s, r/(r+t))
            // E[a] = s*r/(r+t)
            if (a.Precision != 1 || b.Precision != 1)
                throw new NotImplementedException("precision != 1 not implemented");
            double rate = a.Rate / (a.Rate + b.Rate);
            return new Poisson(sum * rate) / a;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/message_doc[@name="BAverageConditional(int, Poisson, Poisson)"]/*'/>
        public static Poisson BAverageConditional(int sum, Poisson a, Poisson b)
        {
            return AAverageConditional(sum, b, a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/message_doc[@name="AAverageConditional(Poisson, Poisson, Poisson)"]/*'/>
        public static Poisson AAverageConditional(Poisson sum, Poisson a, Poisson b)
        {
            if (sum.IsPointMass)
                return AAverageConditional(sum.Point, a, b);
            // p(a) = r^a/a! sum_(s >= a) q^s t^(s-a)/(s-a)!
            //      = r^a/a! q^a sum_(s >= a) q^(s-a) t^(s-a)/(s-a)!
            //      = r^a/a! q^a exp(q t)
            if (sum.Precision != 0)
                throw new NotImplementedException("sum.Precision != 0 not implemented");
            if (b.Precision != 1)
                throw new NotImplementedException("b.Precision != 1 not implemented");
            return new Poisson(sum.Rate, 0);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntegerPlusOp"]/message_doc[@name="BAverageConditional(Poisson, Poisson, Poisson)"]/*'/>
        public static Poisson BAverageConditional(Poisson sum, Poisson a, Poisson b)
        {
            return AAverageConditional(sum, b, a);
        }
    }
}
