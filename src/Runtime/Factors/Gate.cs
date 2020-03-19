// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Factors.Attributes;

namespace Microsoft.ML.Probabilistic.Factors
{
    /// <summary>
    /// Factors for handling gates.
    /// </summary>
    [Hidden]
    public static class Gate
    {

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        /// <summary>
        /// Enter factor
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="selector"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static T[] Enter<T>(int selector, [IsReturnedInEveryElement] T value)
        {
            throw new NotImplementedException();
            T[] result = new T[2];
            for (int i = 0; i < 2; i++)
            {
                result[i] = value;
            }
            return result;
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

        public static T[] Enter<T>(bool selector, [IsReturnedInEveryElement] T value)
        {
            T[] result = new T[2];
            for (int i = 0; i < 2; i++)
            {
                result[i] = value;
            }
            return result;
        }

        /// <summary>
        /// Enter partial factor
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="selector"></param>
        /// <param name="value"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public static T[] EnterPartial<T>(int selector, [IsReturnedInEveryElement] T value, int[] indices)
        {
            T[] result = new T[indices.Length];
            for (int i = 0; i < indices.Length; i++)
            {
                result[i] = value;
            }
            return result;
        }

        public static T[] EnterPartial<T>(bool selector, [IsReturnedInEveryElement] T value, int[] indices)
        {
            T[] result = new T[indices.Length];
            for (int i = 0; i < indices.Length; i++)
            {
                result[i] = value;
            }
            return result;
        }

        /// <summary>
        /// Enter partial factor with two cases
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="case0"></param>
        /// <param name="case1"></param>
        /// <param name="value"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public static T[] EnterPartialTwo<T>(bool case0, bool case1, [IsReturnedInEveryElement] T value, int[] indices)
        {
            T[] result = new T[indices.Length];
            for (int i = 0; i < indices.Length; i++)
            {
                result[i] = value;
            }
            return result;
        }

        /// <summary>
        /// Enter one factor
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="selector"></param>
        /// <param name="value"></param>
        /// <param name="index"></param>
        /// <returns></returns>
        public static T EnterOne<T>(int selector, [IsReturned] T value, int index)
        {
            return value;
        }

        /// <summary>
        /// Exit factor
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="cases"></param>
        /// <param name="values"></param>
        /// <returns></returns>
        public static T Exit<T>(bool[] cases, T[] values)
        {
            for (int i = 0; i < cases.Length; i++)
                if (cases[i])
                    return values[i];

            throw new InferRuntimeException("Exit factor: no case is true");
        }

        /// <summary>
        /// Exit factor with two cases
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="case0"></param>
        /// <param name="case1"></param>
        /// <param name="values"></param>
        /// <returns></returns>
        public static T ExitTwo<T>(bool case0, bool case1, T[] values)
        {
            if (case0)
                return values[0];
            else if (case1)
                return values[1];

            throw new InferRuntimeException("ExitTwo factor: neither case is true");
        }

        /// <summary>
        /// Exit random factor
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="cases"></param>
        /// <param name="values"></param>
        /// <returns></returns>
        [Stochastic]
        [ParameterNames("Exit", "cases", "values")]
        public static T ExitRandom<T>(bool[] cases, T[] values)
        {
            return Exit(cases, values);
        }

        /// <summary>
        /// Exit random factor
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="case0"></param>
        /// <param name="case1"></param>
        /// <param name="values"></param>
        /// <returns></returns>
        [Stochastic]
        [ParameterNames("Exit", "cases", "values")]
        public static T ExitRandomTwo<T>(bool case0, bool case1, T[] values)
        {
            if (case0)
                return values[0];
            else if (case1)
                return values[1];

            throw new InferRuntimeException("ExitTwo factor: neither case is true");
        }

#if true
        /// <summary>
        /// Exiting variable
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="Def"></param>
        /// <param name="Marginal"></param>
        /// <returns></returns>
        [ParameterNames("Use", "Def", "Marginal")]
        public static T ExitingVariable<T>(T Def, out T Marginal)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        /// <summary>
        /// Replicate an exiting variable
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="Def"></param>
        /// <param name="count"></param>
        /// <returns></returns>
        [ParameterNames("Uses", "Def", "count")]
        public static T[] ReplicateExiting<T>(T Def, int count)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }
#else
    /// <summary>
    /// Exiting variable factor
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="Def"></param>
    /// <param name="Marginal"></param>
    /// <returns></returns>
        [ParameterNames("Uses", "Def", "Marginal")]
        public static T[] ExitingVariable<T>(T Def, T Marginal)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }
#endif

        /// <summary>
        /// Boolean cases factor
        /// </summary>
        /// <param name="b"></param>
        /// <returns></returns>
        public static bool[] Cases(bool b)
        {
            bool[] result = new bool[2];
            result[0] = b;
            result[1] = !b;
            return result;
        }

        /// <summary>
        /// Boolean cases factor expanded into elements
        /// </summary>
        /// <param name="b"></param>
        /// <param name="case0">case 0 (true)</param>
        /// <param name="case1">case 1 (false)</param>
        /// <returns></returns>
        public static void CasesBool(bool b, out bool case0, out bool case1)
        {
            case0 = b;
            case1 = !b;
        }

        // TODO: fix bug which prevents this being called 'Cases'
        /// <summary>
        /// Integer cases factor
        /// </summary>
        /// <param name="i">index</param>
        /// <param name="count">number of cases</param>
        /// <returns></returns>
        public static bool[] CasesInt(int i, int count)
        {
            bool[] result = new bool[count];
            for (int j = 0; j < count; j++)
                result[j] = false;
            result[i] = true;
            return result;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CasesOp"]/doc/*'/>
    [FactorMethod(typeof(Gate), "Cases", typeof(bool))]
    [Quality(QualityBand.Mature)]
    public static class CasesOp
    {

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CasesOp"]/message_doc[@name="CasesAverageConditional{BernoulliList}(Bernoulli, BernoulliList)"]/*'/>
        /// <typeparam name="BernoulliList">The type of the outgoing message.</typeparam>
        public static BernoulliList CasesAverageConditional<BernoulliList>(Bernoulli b, BernoulliList result)
            where BernoulliList : IList<Bernoulli>
        {
            // result.LogOdds = [log p(b=true), log p(b=false)]
            if (result.Count != 2)
                throw new ArgumentException("result.Count != 2");
            result[0] = Bernoulli.FromLogOdds(b.GetLogProbTrue());
            result[1] = Bernoulli.FromLogOdds(b.GetLogProbFalse());
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CasesOp"]/message_doc[@name="CasesAverageConditionalInit()"]/*'/>
        [Skip]
        public static DistributionStructArray<Bernoulli, bool> CasesAverageConditionalInit()
        {
            return new DistributionStructArray<Bernoulli, bool>(2);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CasesOp"]/message_doc[@name="BAverageConditional(IList{Bernoulli})"]/*'/>
        public static Bernoulli BAverageConditional([SkipIfUniform] IList<Bernoulli> cases)
        {
            // result = p(b=true) / (p(b=true) + p(b=false))
            //        = 1 / (1 + p(b=false)/p(b=true))
            //        = 1 / (1 + exp(-(log p(b=true) - log p(b=false)))
            // where cases[0].LogOdds = log p(b=true)
            //       cases[1].LogOdds = log p(b=false)
            if (cases[0].LogOdds == cases[1].LogOdds) // avoid (-Infinity) - (-Infinity)
            {
                if (Double.IsNegativeInfinity(cases[0].LogOdds) && Double.IsNegativeInfinity(cases[1].LogOdds))
                    throw new AllZeroException();
                return new Bernoulli();
            }
            else
            {
                return Bernoulli.FromLogOdds(cases[0].LogOdds - cases[1].LogOdds);
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CasesOp"]/message_doc[@name="LogEvidenceRatio(IList{Bernoulli}, Bernoulli)"]/*'/>
        public static double LogEvidenceRatio(IList<Bernoulli> cases, Bernoulli b)
        {
            // result = log (p(data|b=true) p(b=true) + p(data|b=false) p(b=false))
            //          log (p(data|b=true) p(b=true) + p(data|b=false) (1-p(b=true))
            //          log ((p(data|b=true) - p(data|b=false)) p(b=true) + p(data|b=false))
            //          log ((p(data|b=true)/p(data|b=false) - 1) p(b=true) + 1) + log p(data|b=false)
            // where cases[0].LogOdds = log p(data|b=true)
            //       cases[1].LogOdds = log p(data|b=false)
            if (b.IsPointMass)
                return b.Point ? cases[0].LogOdds : cases[1].LogOdds;
            //else return MMath.LogSumExp(cases[0].LogOdds + b.GetLogProbTrue(), cases[1].LogOdds + b.GetLogProbFalse());
            else
            {
                // the common case is when cases[0].LogOdds == cases[1].LogOdds.  we must not introduce rounding error in that case.
                if (cases[0].LogOdds >= cases[1].LogOdds)
                {
                    if (Double.IsNegativeInfinity(cases[1].LogOdds))
                        return cases[0].LogOdds + b.GetLogProbTrue();
                    else
                        return cases[1].LogOdds + MMath.Log1Plus(b.GetProbTrue() * MMath.ExpMinus1(cases[0].LogOdds - cases[1].LogOdds));
                }
                else
                {
                    if (Double.IsNegativeInfinity(cases[0].LogOdds))
                        return cases[1].LogOdds + b.GetLogProbFalse();
                    else
                        return cases[0].LogOdds + MMath.Log1Plus(b.GetProbFalse() * MMath.ExpMinus1(cases[1].LogOdds - cases[0].LogOdds));
                }
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CasesOp"]/message_doc[@name="LogEvidenceRatio(IList{Bernoulli}, bool)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(IList<Bernoulli> cases, bool b)
        {
            return 0.0;
            //return b ? cases[0].LogOdds : cases[1].LogOdds;
        }

        //-- VMP --------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CasesOp"]/message_doc[@name="CasesAverageLogarithm{BernoulliList}(Bernoulli, BernoulliList)"]/*'/>
        /// <typeparam name="BernoulliList">The type of the outgoing message.</typeparam>
        public static BernoulliList CasesAverageLogarithm<BernoulliList>(Bernoulli b, BernoulliList result)
            where BernoulliList : IList<Bernoulli>
        {
            return CasesAverageConditional(b, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CasesOp"]/message_doc[@name="CasesAverageLogarithmInit()"]/*'/>
        [Skip]
        public static DistributionStructArray<Bernoulli, bool> CasesAverageLogarithmInit()
        {
            return new DistributionStructArray<Bernoulli, bool>(2);
        }

        [Skip]
        public static DistributionType CasesDeriv<DistributionType>(DistributionType result)
            where DistributionType : SettableToUniform
        {
            result.SetToUniform();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CasesOp"]/message_doc[@name="BAverageLogarithm(IList{Bernoulli})"]/*'/>
        public static Bernoulli BAverageLogarithm([SkipIfUniform] IList<Bernoulli> cases) // TM: SkipIfAny (rather than SkipIfAll) is important for getting good schedules
        {
            return BAverageConditional(cases);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CasesOp"]/message_doc[@name="AverageLogFactor(IList{Bernoulli}, Bernoulli)"]/*'/>
        public static double AverageLogFactor([SkipIfUniform] IList<Bernoulli> cases, Bernoulli b)
        {
            double probTrue = b.GetProbTrue();
            return probTrue * cases[0].LogOdds + (1 - probTrue) * cases[1].LogOdds;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CasesBoolOp"]/doc/*'/>
    [FactorMethod(typeof(Gate), "CasesBool", typeof(bool), typeof(bool), typeof(bool))]
    [Quality(QualityBand.Experimental)]
    public static class CasesBoolOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CasesBoolOp"]/message_doc[@name="Case0AverageConditional(Bernoulli)"]/*'/>
        public static Bernoulli Case0AverageConditional(Bernoulli b)
        {
            return Bernoulli.FromLogOdds(b.GetLogProbTrue());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CasesBoolOp"]/message_doc[@name="Case1AverageConditional(Bernoulli)"]/*'/>
        public static Bernoulli Case1AverageConditional(Bernoulli b)
        {
            return Bernoulli.FromLogOdds(b.GetLogProbFalse());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CasesBoolOp"]/message_doc[@name="BAverageConditional(Bernoulli, Bernoulli)"]/*'/>
        [SkipIfAllUniform]
        public static Bernoulli BAverageConditional(Bernoulli case0, Bernoulli case1)
        {
            // result = p(b=true) / (p(b=true) + p(b=false))
            //        = 1 / (1 + p(b=false)/p(b=true))
            //        = 1 / (1 + exp(-(log p(b=true) - log p(b=false)))
            // where cases[0].LogOdds = log p(b=true)
            //       cases[1].LogOdds = log p(b=false)
            // avoid (-Infinity) - (-Infinity)
            if (case0.LogOdds == case1.LogOdds)
            {
                if (Double.IsNegativeInfinity(case0.LogOdds) && Double.IsNegativeInfinity(case1.LogOdds))
                    throw new AllZeroException();
                return new Bernoulli();
            }
            else
            {
                return Bernoulli.FromLogOdds(case0.LogOdds - case1.LogOdds);
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CasesBoolOp"]/message_doc[@name="LogEvidenceRatio(Bernoulli, Bernoulli, Bernoulli)"]/*'/>
        public static double LogEvidenceRatio(Bernoulli case0, Bernoulli case1, Bernoulli b)
        {
            // result = log (p(data|b=true) p(b=true) + p(data|b=false) p(b=false))
            // where cases[0].LogOdds = log p(data|b=true)
            //       cases[1].LogOdds = log p(data|b=false)
            if (b.IsPointMass)
                return b.Point ? case0.LogOdds : case1.LogOdds;
            else
                return MMath.LogSumExp(case0.LogOdds + b.GetLogProbTrue(), case1.LogOdds + b.GetLogProbFalse());
        }

        //-- VMP --------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CasesBoolOp"]/message_doc[@name="Case0AverageLogarithm(Bernoulli)"]/*'/>
        public static Bernoulli Case0AverageLogarithm(Bernoulli b)
        {
            return Case0AverageConditional(b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CasesBoolOp"]/message_doc[@name="Case1AverageLogarithm(Bernoulli)"]/*'/>
        public static Bernoulli Case1AverageLogarithm(Bernoulli b)
        {
            return Case1AverageConditional(b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CasesBoolOp"]/message_doc[@name="BAverageLogarithm(Bernoulli, Bernoulli)"]/*'/>
        [SkipIfAllUniform]
        public static Bernoulli BAverageLogarithm(Bernoulli case0, Bernoulli case1)
        {
            return BAverageConditional(case0, case1);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CasesBoolOp"]/message_doc[@name="AverageLogFactor(Bernoulli, Bernoulli, Bernoulli)"]/*'/>
        [SkipIfAllUniform("case0", "case1")]
        public static double AverageLogFactor(Bernoulli case0, Bernoulli case1, Bernoulli b)
        {
            double probTrue = b.GetProbTrue();
            return probTrue * case0.LogOdds + (1 - probTrue) * case1.LogOdds;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntCasesOp"]/doc/*'/>
    [FactorMethod(new string[] { "Cases", "i", "count" }, typeof(Gate), "CasesInt", typeof(int), typeof(int))]
    [Quality(QualityBand.Mature)]
    public static class IntCasesOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntCasesOp"]/message_doc[@name="CasesAverageConditional(Discrete, int)"]/*'/>
        public static Bernoulli CasesAverageConditional(Discrete i, int resultIndex)
        {
            return Bernoulli.FromLogOdds(i.GetLogProb(resultIndex));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntCasesOp"]/message_doc[@name="IAverageConditional(IList{Bernoulli}, Discrete)"]/*'/>
        public static Discrete IAverageConditional([SkipIfUniform] IList<Bernoulli> cases, Discrete result)
        {
            Vector probs = result.GetWorkspace();
            double max = cases[0].LogOdds;
            for (int j = 1; j < cases.Count; j++)
            {
                if (cases[j].LogOdds > max)
                    max = cases[j].LogOdds;
            }
            // if result.Dimension > cases.Count, the missing cases have LogOdds=0
            if (result.Dimension > cases.Count)
                max = System.Math.Max(max, 0);
            // avoid (-Infinity) - (-Infinity)
            if (Double.IsNegativeInfinity(max))
                throw new AllZeroException();

            if (probs.Sparsity.IsApproximate)
            {
                var sparseProbs = probs as ApproximateSparseVector;
                probs.SetAllElementsTo(sparseProbs.Tolerance);
            }

            for (int j = 0; j < result.Dimension; j++)
            {
                if (j < cases.Count)
                    probs[j] = System.Math.Exp(cases[j].LogOdds - max);
                else
                    probs[j] = System.Math.Exp(0 - max);
            }

            result.SetProbs(probs);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntCasesOp"]/message_doc[@name="LogEvidenceRatio(IList{Bernoulli}, Discrete)"]/*'/>
        public static double LogEvidenceRatio([SkipIfUniform] IList<Bernoulli> cases, Discrete i)
        {
            if (i.IsPointMass)
                return cases[i.Point].LogOdds;
            else
            {
                double[] logOdds = new double[cases.Count];
                for (int j = 0; j < cases.Count; j++)
                {
                    logOdds[j] = cases[j].LogOdds + i.GetLogProb(j);
                }
                return MMath.LogSumExp(logOdds);
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntCasesOp"]/message_doc[@name="LogEvidenceRatio(IList{Bernoulli}, int)"]/*'/>
        public static double LogEvidenceRatio(IList<Bernoulli> cases, int i)
        {
            return cases[i].LogOdds;
        }

        //-- VMP --------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntCasesOp"]/message_doc[@name="CasesAverageLogarithm(Discrete, int)"]/*'/>
        public static Bernoulli CasesAverageLogarithm(Discrete i, int resultIndex)
        {
            return CasesAverageConditional(i, resultIndex);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntCasesOp"]/message_doc[@name="CasesAverageLogarithmInit(Discrete)"]/*'/>
        [Skip]
        public static DistributionStructArray<Bernoulli, bool> CasesAverageLogarithmInit([IgnoreDependency] Discrete i)
        {
            return new DistributionStructArray<Bernoulli, bool>(i.Dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntCasesOp"]/message_doc[@name="IAverageLogarithm(IList{Bernoulli}, Discrete)"]/*'/>
        public static Discrete IAverageLogarithm([SkipIfUniform] IList<Bernoulli> cases, Discrete result)
        {
            return IAverageConditional(cases, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IntCasesOp"]/message_doc[@name="AverageLogFactor(IList{Bernoulli}, Discrete)"]/*'/>
        public static double AverageLogFactor([SkipIfAllUniform] IList<Bernoulli> cases, Discrete i)
        {
            if (i.IsPointMass)
                return cases[i.Point].LogOdds;  // avoid possible 0 * infinity below
            double sum = 0.0;
            for (int j = 0; j < cases.Count; j++)
            {
                sum += i[j] * cases[j].LogOdds;
            }
            return sum;
        }
    }
}
