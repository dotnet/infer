// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Collections.Generic;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <summary>
    /// A repository of commonly used constraint methods.
    /// </summary>
    public static class Constrain
    {
        /// <summary>
        /// Constrains a value to be equal to a sample from dist.
        /// </summary>
        /// <typeparam name="TDomain">Domain type</typeparam>
        /// <typeparam name="TDistribution">Distribution type</typeparam>
        /// <param name="value">Value</param>
        /// <param name="dist">Distribution instance</param>
        [Stochastic]
        public static void EqualRandom<TDomain, TDistribution>(TDomain value, TDistribution dist)
            where TDistribution : Sampleable<TDomain>
        {
            IEqualityComparer<TDomain> equalityComparer = Utilities.Util.GetEqualityComparer<TDomain>();
            if (!equalityComparer.Equals(value, dist.Sample()))
                throw new ConstraintViolatedException(value + " != " + dist);
        }

        /// <summary>
        /// Constrains a value to be equal to another value.
        /// </summary>
        /// <typeparam name="T">Value type</typeparam>
        /// <param name="A">First value</param>
        /// <param name="B">Second value</param>
        public static void Equal<T>(T A, T B)
        {
            IEqualityComparer<T> equalityComparer = Utilities.Util.GetEqualityComparer<T>();
            if (!equalityComparer.Equals(A,B))
                throw new ConstraintViolatedException(A + " != " + B);
        }

        /// <summary>
        /// Constrains a set of integers to contain a particular integer.
        /// </summary>
        /// <param name="set">The set of integers, specified as a list</param>
        /// <param name="i">The integer which the set must contain</param>
        [Hidden]
        public static void Contain(IList<int> set, int i)
        {
            if (!set.Contains(i))
            {
                throw new ConstraintViolatedException(
                    "Containment constraint violated (the supplied set did not contain the integer " + i + ")");
            }
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConstrainEqualRandomOp{TDomain}"]/doc/*'/>
    /// <typeparam name="TDomain">The domain of the constrained variables.</typeparam>
    [FactorMethod(typeof(Constrain), "EqualRandom<,>")]
    [Quality(QualityBand.Mature)]
    public static class ConstrainEqualRandomOp<TDomain>    
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConstrainEqualRandomOp{TDomain}"]/message_doc[@name="ValueAverageConditional{TDistribution}(TDistribution)"]/*'/>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static TDistribution ValueAverageConditional<TDistribution>([IsReturned] TDistribution dist)
        {
            return dist;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConstrainEqualRandomOp{TDomain}"]/message_doc[@name="LogAverageFactor{TDistribution}(TDistribution, TDistribution)"]/*'/>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static double LogAverageFactor<TDistribution>(TDistribution value, TDistribution dist)
            where TDistribution : CanGetLogAverageOf<TDistribution>
        {
            return value.GetLogAverageOf(dist);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConstrainEqualRandomOp{TDomain}"]/message_doc[@name="LogAverageFactor{TDistribution}(TDomain, TDistribution)"]/*'/>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static double LogAverageFactor<TDistribution>(TDomain value, [Proper] TDistribution dist)
            where TDistribution : CanGetLogProb<TDomain>
        {
            return dist.GetLogProb(value);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConstrainEqualRandomOp{TDomain}"]/message_doc[@name="LogEvidenceRatio{TDistribution}(TDistribution, TDistribution)"]/*'/>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static double LogEvidenceRatio<TDistribution>(TDistribution value, TDistribution dist)
            where TDistribution : CanGetLogAverageOf<TDistribution>
        {
            return LogAverageFactor(value, dist);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConstrainEqualRandomOp{TDomain}"]/message_doc[@name="LogEvidenceRatio{TDistribution}(TDomain, TDistribution)"]/*'/>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static double LogEvidenceRatio<TDistribution>(TDomain value, TDistribution dist)
            where TDistribution : CanGetLogProb<TDomain>
        {
            return LogAverageFactor(value, dist);
        }

        //-- VMP --------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConstrainEqualRandomOp{TDomain}"]/message_doc[@name="AverageLogFactor{TDistribution}(TDistribution, TDistribution)"]/*'/>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static double AverageLogFactor<TDistribution>(TDistribution value, TDistribution dist)
            where TDistribution : CanGetAverageLog<TDistribution>
        {
            return value.GetAverageLog(dist);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConstrainEqualRandomOp{TDomain}"]/message_doc[@name="AverageLogFactor{TDistribution}(TDomain, TDistribution)"]/*'/>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static double AverageLogFactor<TDistribution>(TDomain value, [Proper] TDistribution dist)
            where TDistribution : CanGetLogProb<TDomain>
        {
            return dist.GetLogProb(value);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConstrainEqualRandomOp{TDomain}"]/message_doc[@name="ValueAverageLogarithm{TDistribution}(TDistribution)"]/*'/>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static TDistribution ValueAverageLogarithm<TDistribution>([IsReturned] TDistribution dist)
        {
            return dist;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConstrainEqualOp{T}"]/doc/*'/>
    /// <typeparam name="T">The type of the constrained variables.</typeparam>
    [FactorMethod(typeof(Constrain), "Equal<>")]
    [Quality(QualityBand.Mature)]
    public static class ConstrainEqualOp<T>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConstrainEqualOp{T}"]/message_doc[@name="LogAverageFactor{TDistribution}(TDistribution, TDistribution)"]/*'/>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static double LogAverageFactor<TDistribution>(TDistribution a, TDistribution b)
            where TDistribution : CanGetLogAverageOf<TDistribution>
        {
            return a.GetLogAverageOf(b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConstrainEqualOp{T}"]/message_doc[@name="LogAverageFactor{TDistribution}(T, TDistribution)"]/*'/>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static double LogAverageFactor<TDistribution>(T a, TDistribution b)
            where TDistribution : CanGetLogProb<T>
        {
            return b.GetLogProb(a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConstrainEqualOp{T}"]/message_doc[@name="LogAverageFactor{TDistribution}(TDistribution, T)"]/*'/>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static double LogAverageFactor<TDistribution>(TDistribution a, T b)
            where TDistribution : CanGetLogProb<T>
        {
            return a.GetLogProb(b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConstrainEqualOp{T}"]/message_doc[@name="LogAverageFactor(T, T)"]/*'/>
        public static double LogAverageFactor(T a, T b)
        {
            IEqualityComparer<T> equalityComparer = Utilities.Util.GetEqualityComparer<T>();
            return (equalityComparer.Equals(a, b) ? 0.0 : Double.NegativeInfinity);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConstrainEqualOp{T}"]/message_doc[@name="LogEvidenceRatio{TDistribution}(TDistribution, TDistribution)"]/*'/>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static double LogEvidenceRatio<TDistribution>(TDistribution a, TDistribution b)
            where TDistribution : CanGetLogAverageOf<TDistribution>
        {
            return LogAverageFactor(a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConstrainEqualOp{T}"]/message_doc[@name="LogEvidenceRatio{TDistribution}(T, TDistribution)"]/*'/>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static double LogEvidenceRatio<TDistribution>(T a, TDistribution b)
            where TDistribution : CanGetLogProb<T>
        {
            return LogAverageFactor(a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConstrainEqualOp{T}"]/message_doc[@name="LogEvidenceRatio{TDistribution}(TDistribution, T)"]/*'/>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static double LogEvidenceRatio<TDistribution>(TDistribution a, T b)
            where TDistribution : CanGetLogProb<T>
        {
            return LogAverageFactor(a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConstrainEqualOp{T}"]/message_doc[@name="LogEvidenceRatio(T, T)"]/*'/>
        public static double LogEvidenceRatio(T a, T b)
        {
            return LogAverageFactor(a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConstrainEqualOp{T}"]/message_doc[@name="AAverageConditional{TDistribution}(TDistribution, TDistribution)"]/*'/>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static TDistribution AAverageConditional<TDistribution>([IsReturned] TDistribution B, TDistribution result)
            where TDistribution : SettableTo<TDistribution>
        {
            result.SetTo(B);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConstrainEqualOp{T}"]/message_doc[@name="AAverageConditional{TDistribution}(T, TDistribution)"]/*'/>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static TDistribution AAverageConditional<TDistribution>(T B, TDistribution result)
            where TDistribution : HasPoint<T>
        {
            result.Point = B;
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConstrainEqualOp{T}"]/message_doc[@name="BAverageConditional{TDistribution}(TDistribution, TDistribution)"]/*'/>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static TDistribution BAverageConditional<TDistribution>([IsReturned] TDistribution A, TDistribution result)
            where TDistribution : SettableTo<TDistribution>
        {
            result.SetTo(A);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConstrainEqualOp{T}"]/message_doc[@name="BAverageConditional{TDistribution}(T, TDistribution)"]/*'/>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static TDistribution BAverageConditional<TDistribution>(T A, TDistribution result)
            where TDistribution : HasPoint<T>
        {
            return AAverageConditional<TDistribution>(A, result);
        }

        //-- VMP -----------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConstrainEqualOp{T}"]/message_doc[@name="AverageLogFactor()"]/*'/>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        private const string NotSupportedMessage = "VMP does not support Constrain.Equal between random variables";

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConstrainEqualOp{T}"]/message_doc[@name="AAverageLogarithm{TDistribution}(TDistribution, TDistribution)"]/*'/>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        [NotSupported(NotSupportedMessage)]
        public static TDistribution AAverageLogarithm<TDistribution>(TDistribution B, TDistribution result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConstrainEqualOp{T}"]/message_doc[@name="BAverageLogarithm{TDistribution}(TDistribution, TDistribution)"]/*'/>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        [NotSupported(NotSupportedMessage)]
        public static TDistribution BAverageLogarithm<TDistribution>(TDistribution A, TDistribution result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConstrainEqualOp{T}"]/message_doc[@name="AAverageLogarithm{TDistribution}(T, TDistribution)"]/*'/>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static TDistribution AAverageLogarithm<TDistribution>(T B, TDistribution result)
            where TDistribution : HasPoint<T>
        {
            result.Point = B;
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConstrainEqualOp{T}"]/message_doc[@name="BAverageLogarithm{TDistribution}(T, TDistribution)"]/*'/>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static TDistribution BAverageLogarithm<TDistribution>(T A, TDistribution result)
            where TDistribution : HasPoint<T>
        {
            return AAverageLogarithm<TDistribution>(A, result);
        }

        //-- Max product ----------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConstrainEqualOp{T}"]/message_doc[@name="AMaxConditional{TDistribution}(TDistribution, TDistribution)"]/*'/>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static TDistribution AMaxConditional<TDistribution>([IsReturned] TDistribution B, TDistribution result)
            where TDistribution : SettableTo<TDistribution>
        {
            result.SetTo(B);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConstrainEqualOp{T}"]/message_doc[@name="BMaxConditional{TDistribution}(TDistribution, TDistribution)"]/*'/>
        /// <typeparam name="TDistribution">The distribution over the constrained variables.</typeparam>
        public static TDistribution BMaxConditional<TDistribution>([IsReturned] TDistribution A, TDistribution result)
            where TDistribution : SettableTo<TDistribution>
        {
            result.SetTo(A);
            return result;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConstrainContainOp"]/doc/*'/>
    [FactorMethod(typeof(Constrain), "Contain")]
    [Quality(QualityBand.Experimental)]
    public class ConstrainContainOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ConstrainContainOp"]/message_doc[@name="SetAverageLogarithm(int, BernoulliIntegerSubset)"]/*'/>
        public static BernoulliIntegerSubset SetAverageLogarithm(int i, BernoulliIntegerSubset result)
        {
            result.SetToUniform();
            result.SparseBernoulliList[i] = Bernoulli.PointMass(true);
            return result;
        }
    }
}
