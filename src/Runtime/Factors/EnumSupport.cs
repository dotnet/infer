// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Factors.Attributes;

namespace Microsoft.ML.Probabilistic.Factors
{
    /// <summary>
    /// Provides factors and operators for using Enum types.
    /// </summary>
    public class EnumSupport
    {
        /// <summary>
        /// Converts an Enum to an Int
        /// </summary>
        /// <param name="en"></param>
        /// <returns></returns>
        [ParameterNames("Int", "Enum")]
        public static int EnumToInt<TEnum>(TEnum en)
        {
            return (int)(object)en;
        }

        /// <summary>
        /// Samples an enum value from a discrete enum distribution.
        /// </summary>
        /// <typeparam name="TEnum">The type of the enum to sample</typeparam>
        /// <param name="probs">Vector of the probability of each Enum value, in order</param>
        /// <returns>An enum sampled from the distribution</returns>
        [Stochastic]
        [ParameterNames("Sample", "Probs")]
        public static TEnum DiscreteEnum<TEnum>(Vector probs)
        {
            int i = Microsoft.ML.Probabilistic.Distributions.Discrete.Sample(probs);
            return (TEnum)Enum.GetValues(typeof(TEnum)).GetValue(i);
        }

        /// <summary>
        /// Test if two enums are equal.
        /// </summary>
        /// <param name="a">First integer</param>
        /// <param name="b">Second integer</param>
        /// <returns>True if a==b.</returns>
        public static bool AreEqual<TEnum>(TEnum a, TEnum b)
        {
            return EnumToInt<TEnum>(a) == EnumToInt<TEnum>(b);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="EnumToIntOp{TEnum}"]/doc/*'/>
    /// <typeparam name="TEnum">The type of the enumeration.</typeparam>
    [FactorMethod(typeof(EnumSupport), "EnumToInt<>")]
    [Quality(QualityBand.Preview)]
    public static class EnumToIntOp<TEnum>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="EnumToIntOp{TEnum}"]/message_doc[@name="LogAverageFactor(int, TEnum)"]/*'/>
        public static double LogAverageFactor(int Int, TEnum Enum)
        {
            return (EnumSupport.EnumToInt(Enum) == Int) ? 0.0 : Double.NegativeInfinity;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="EnumToIntOp{TEnum}"]/message_doc[@name="LogEvidenceRatio(int, TEnum)"]/*'/>
        public static double LogEvidenceRatio(int Int, TEnum Enum)
        {
            return LogAverageFactor(Int, Enum);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="EnumToIntOp{TEnum}"]/message_doc[@name="AverageLogFactor(int, TEnum)"]/*'/>
        public static double AverageLogFactor(int Int, TEnum Enum)
        {
            return LogAverageFactor(Int, Enum);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="EnumToIntOp{TEnum}"]/message_doc[@name="LogAverageFactor(int, DiscreteEnum{TEnum})"]/*'/>
        public static double LogAverageFactor(int Int, DiscreteEnum<TEnum> Enum)
        {
            return Enum.GetLogProb((TEnum)(object)Int);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="EnumToIntOp{TEnum}"]/message_doc[@name="LogEvidenceRatio(int, DiscreteEnum{TEnum})"]/*'/>
        public static double LogEvidenceRatio(int Int, DiscreteEnum<TEnum> Enum)
        {
            return LogAverageFactor(Int, Enum);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="EnumToIntOp{TEnum}"]/message_doc[@name="LogAverageFactor(Discrete, TEnum)"]/*'/>
        public static double LogAverageFactor(Discrete Int, TEnum Enum)
        {
            return Int.GetLogProb((int)(object)Enum);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="EnumToIntOp{TEnum}"]/message_doc[@name="LogAverageFactor(Discrete, DiscreteEnum{TEnum}, Discrete)"]/*'/>
        public static double LogAverageFactor(Discrete Int, DiscreteEnum<TEnum> Enum, [Fresh] Discrete to_Int)
        {
            return to_Int.GetLogAverageOf(Int);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="EnumToIntOp{TEnum}"]/message_doc[@name="LogEvidenceRatio(Discrete)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Discrete Int)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="EnumToIntOp{TEnum}"]/message_doc[@name="AverageLogFactor()"]/*'/>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="EnumToIntOp{TEnum}"]/message_doc[@name="IntAverageConditional(DiscreteEnum{TEnum}, Discrete)"]/*'/>
        public static Discrete IntAverageConditional([SkipIfUniform] DiscreteEnum<TEnum> Enum, Discrete result)
        {
            result.SetProbs(Enum.GetWorkspace());
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="EnumToIntOp{TEnum}"]/message_doc[@name="IntAverageConditionalInit(DiscreteEnum{TEnum})"]/*'/>
        [Skip]
        public static Discrete IntAverageConditionalInit([IgnoreDependency] DiscreteEnum<TEnum> Enum)
        {
            return Discrete.Uniform(Enum.Dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="EnumToIntOp{TEnum}"]/message_doc[@name="EnumAverageConditional(Discrete, DiscreteEnum{TEnum})"]/*'/>
        public static DiscreteEnum<TEnum> EnumAverageConditional([SkipIfUniform] Discrete Int, DiscreteEnum<TEnum> result)
        {
            result.SetProbs(Int.GetWorkspace());
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="EnumToIntOp{TEnum}"]/message_doc[@name="EnumAverageConditional(int, DiscreteEnum{TEnum})"]/*'/>
        public static DiscreteEnum<TEnum> EnumAverageConditional(int Int, DiscreteEnum<TEnum> result)
        {
            result.Point = (TEnum)Enum.GetValues(typeof(TEnum)).GetValue(Int);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="EnumToIntOp{TEnum}"]/message_doc[@name="IntAverageLogarithm(DiscreteEnum{TEnum}, Discrete)"]/*'/>
        public static Discrete IntAverageLogarithm([SkipIfUniform] DiscreteEnum<TEnum> Enum, Discrete result)
        {
            return IntAverageConditional(Enum, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="EnumToIntOp{TEnum}"]/message_doc[@name="EnumAverageLogarithm(Discrete, DiscreteEnum{TEnum})"]/*'/>
        public static DiscreteEnum<TEnum> EnumAverageLogarithm([SkipIfUniform] Discrete Int, DiscreteEnum<TEnum> result)
        {
            return EnumAverageConditional(Int, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="EnumToIntOp{TEnum}"]/message_doc[@name="EnumAverageLogarithm(int, DiscreteEnum{TEnum})"]/*'/>
        public static DiscreteEnum<TEnum> EnumAverageLogarithm(int Int, DiscreteEnum<TEnum> result)
        {
            return EnumAverageConditional(Int, result);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumFromDirichletOp{TEnum}"]/doc/*'/>
    /// <typeparam name="TEnum">The type of the enumeration.</typeparam>
    /// <remarks>
    /// This class provides operators which have <see cref="Enum"/> arguments.  
    /// The rest are provided by <see cref="DiscreteFromDirichletOp"/>.
    /// </remarks>
    [FactorMethod(typeof(EnumSupport), "DiscreteEnum<>")]
    [Quality(QualityBand.Stable)]
    public static class DiscreteEnumFromDirichletOp<TEnum>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumFromDirichletOp{TEnum}"]/message_doc[@name="LogAverageFactor(TEnum, Dirichlet)"]/*'/>
        public static double LogAverageFactor(TEnum sample, Dirichlet probs)
        {
            return DiscreteFromDirichletOp.LogAverageFactor(EnumSupport.EnumToInt(sample), probs);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumFromDirichletOp{TEnum}"]/message_doc[@name="LogAverageFactor(TEnum, Vector)"]/*'/>
        public static double LogAverageFactor(TEnum sample, Vector probs)
        {
            return DiscreteFromDirichletOp.LogAverageFactor(EnumSupport.EnumToInt(sample), probs);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumFromDirichletOp{TEnum}"]/message_doc[@name="AverageLogFactor(TEnum, Dirichlet)"]/*'/>
        public static double AverageLogFactor(TEnum sample, Dirichlet probs)
        {
            return DiscreteFromDirichletOp.AverageLogFactor(EnumSupport.EnumToInt(sample), probs);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumFromDirichletOp{TEnum}"]/message_doc[@name="AverageLogFactor(TEnum, Vector)"]/*'/>
        public static double AverageLogFactor(TEnum sample, Vector probs)
        {
            return DiscreteFromDirichletOp.AverageLogFactor(EnumSupport.EnumToInt(sample), probs);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumFromDirichletOp{TEnum}"]/message_doc[@name="LogEvidenceRatio(TEnum, Dirichlet)"]/*'/>
        public static double LogEvidenceRatio(TEnum sample, Dirichlet probs)
        {
            return DiscreteFromDirichletOp.LogEvidenceRatio(EnumSupport.EnumToInt(sample), probs);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumFromDirichletOp{TEnum}"]/message_doc[@name="LogEvidenceRatio(TEnum, Vector)"]/*'/>
        public static double LogEvidenceRatio(TEnum sample, Vector probs)
        {
            return DiscreteFromDirichletOp.LogEvidenceRatio(EnumSupport.EnumToInt(sample), probs);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumFromDirichletOp{TEnum}"]/message_doc[@name="ProbsAverageConditional(TEnum, Dirichlet)"]/*'/>
        public static Dirichlet ProbsAverageConditional(TEnum sample, Dirichlet result)
        {
            return DiscreteFromDirichletOp.ProbsAverageConditional(EnumSupport.EnumToInt(sample), result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumFromDirichletOp{TEnum}"]/message_doc[@name="ProbsAverageLogarithm(TEnum, Dirichlet)"]/*'/>
        public static Dirichlet ProbsAverageLogarithm(TEnum sample, Dirichlet result)
        {
            return DiscreteFromDirichletOp.ProbsAverageLogarithm(EnumSupport.EnumToInt(sample), result);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumAreEqualOp{TEnum}"]/doc/*'/>
    /// <typeparam name="TEnum">The type of the enumeration.</typeparam>
    /// <remarks>
    /// This class only implements enumeration-specific overloads that are not provided by <see cref="DiscreteAreEqualOp"/>.
    /// </remarks>
    [FactorMethod(typeof(EnumSupport), "AreEqual<>")]
    [Quality(QualityBand.Stable)]
    public class DiscreteEnumAreEqualOp<TEnum>
    {
        private static int ToInt(TEnum en)
        {
            return (int)(object)en;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumAreEqualOp{TEnum}"]/message_doc[@name="AAverageConditional(Bernoulli, DiscreteEnum{TEnum}, DiscreteEnum{TEnum})"]/*'/>
        public static DiscreteEnum<TEnum> AAverageConditional([SkipIfUniform] Bernoulli areEqual, DiscreteEnum<TEnum> B, DiscreteEnum<TEnum> result)
        {
            return DiscreteEnum<TEnum>.FromDiscrete(DiscreteAreEqualOp.AAverageConditional(areEqual, B.GetInternalDiscrete(), result.GetInternalDiscrete()));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumAreEqualOp{TEnum}"]/message_doc[@name="AAverageConditional(Bernoulli, TEnum, DiscreteEnum{TEnum})"]/*'/>
        public static DiscreteEnum<TEnum> AAverageConditional([SkipIfUniform] Bernoulli areEqual, TEnum B, DiscreteEnum<TEnum> result)
        {
            return DiscreteEnum<TEnum>.FromDiscrete(DiscreteAreEqualOp.AAverageConditional(areEqual, ToInt(B), result.GetInternalDiscrete()));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumAreEqualOp{TEnum}"]/message_doc[@name="AAverageConditional(bool, TEnum, DiscreteEnum{TEnum})"]/*'/>
        public static DiscreteEnum<TEnum> AAverageConditional(bool areEqual, TEnum B, DiscreteEnum<TEnum> result)
        {
            return DiscreteEnum<TEnum>.FromDiscrete(DiscreteAreEqualOp.AAverageConditional(areEqual, ToInt(B), result.GetInternalDiscrete()));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumAreEqualOp{TEnum}"]/message_doc[@name="AAverageLogarithm(Bernoulli, DiscreteEnum{TEnum}, DiscreteEnum{TEnum})"]/*'/>
        public static DiscreteEnum<TEnum> AAverageLogarithm([SkipIfUniform] Bernoulli areEqual, DiscreteEnum<TEnum> B, DiscreteEnum<TEnum> result)
        {
            return DiscreteEnum<TEnum>.FromDiscrete(DiscreteAreEqualOp.AAverageLogarithm(areEqual, B.GetInternalDiscrete(), result.GetInternalDiscrete()));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumAreEqualOp{TEnum}"]/message_doc[@name="AAverageLogarithm(bool, TEnum, DiscreteEnum{TEnum})"]/*'/>
        public static DiscreteEnum<TEnum> AAverageLogarithm(bool areEqual, TEnum B, DiscreteEnum<TEnum> result)
        {
            return DiscreteEnum<TEnum>.FromDiscrete(DiscreteAreEqualOp.AAverageLogarithm(areEqual, ToInt(B), result.GetInternalDiscrete()));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumAreEqualOp{TEnum}"]/message_doc[@name="AAverageLogarithm(Bernoulli, TEnum, DiscreteEnum{TEnum})"]/*'/>
        public static DiscreteEnum<TEnum> AAverageLogarithm([SkipIfUniform] Bernoulli areEqual, TEnum B, DiscreteEnum<TEnum> result)
        {
            return DiscreteEnum<TEnum>.FromDiscrete(DiscreteAreEqualOp.AAverageLogarithm(areEqual, ToInt(B), result.GetInternalDiscrete()));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumAreEqualOp{TEnum}"]/message_doc[@name="BAverageConditional(Bernoulli, DiscreteEnum{TEnum}, DiscreteEnum{TEnum})"]/*'/>
        public static DiscreteEnum<TEnum> BAverageConditional([SkipIfUniform] Bernoulli areEqual, DiscreteEnum<TEnum> A, DiscreteEnum<TEnum> result)
        {
            return DiscreteEnum<TEnum>.FromDiscrete(DiscreteAreEqualOp.BAverageConditional(areEqual, A.GetInternalDiscrete(), result.GetInternalDiscrete()));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumAreEqualOp{TEnum}"]/message_doc[@name="BAverageConditional(Bernoulli, TEnum, DiscreteEnum{TEnum})"]/*'/>
        public static DiscreteEnum<TEnum> BAverageConditional([SkipIfUniform] Bernoulli areEqual, TEnum A, DiscreteEnum<TEnum> result)
        {
            return DiscreteEnum<TEnum>.FromDiscrete(DiscreteAreEqualOp.BAverageConditional(areEqual, ToInt(A), result.GetInternalDiscrete()));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumAreEqualOp{TEnum}"]/message_doc[@name="BAverageConditional(bool, TEnum, DiscreteEnum{TEnum})"]/*'/>
        public static DiscreteEnum<TEnum> BAverageConditional(bool areEqual, TEnum A, DiscreteEnum<TEnum> result)
        {
            return DiscreteEnum<TEnum>.FromDiscrete(DiscreteAreEqualOp.BAverageConditional(areEqual, ToInt(A), result.GetInternalDiscrete()));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumAreEqualOp{TEnum}"]/message_doc[@name="BAverageLogarithm(Bernoulli, DiscreteEnum{TEnum}, DiscreteEnum{TEnum})"]/*'/>
        public static DiscreteEnum<TEnum> BAverageLogarithm([SkipIfUniform] Bernoulli areEqual, DiscreteEnum<TEnum> A, DiscreteEnum<TEnum> result)
        {
            return DiscreteEnum<TEnum>.FromDiscrete(DiscreteAreEqualOp.BAverageLogarithm(areEqual, A.GetInternalDiscrete(), result.GetInternalDiscrete()));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumAreEqualOp{TEnum}"]/message_doc[@name="BAverageLogarithm(Bernoulli, TEnum, DiscreteEnum{TEnum})"]/*'/>
        public static DiscreteEnum<TEnum> BAverageLogarithm([SkipIfUniform] Bernoulli areEqual, TEnum A, DiscreteEnum<TEnum> result)
        {
            return DiscreteEnum<TEnum>.FromDiscrete(DiscreteAreEqualOp.BAverageLogarithm(areEqual, ToInt(A), result.GetInternalDiscrete()));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumAreEqualOp{TEnum}"]/message_doc[@name="BAverageLogarithm(bool, TEnum, DiscreteEnum{TEnum})"]/*'/>
        public static DiscreteEnum<TEnum> BAverageLogarithm(bool areEqual, TEnum A, DiscreteEnum<TEnum> result)
        {
            return DiscreteEnum<TEnum>.FromDiscrete(DiscreteAreEqualOp.BAverageLogarithm(areEqual, ToInt(A), result.GetInternalDiscrete()));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumAreEqualOp{TEnum}"]/message_doc[@name="AreEqualAverageConditional(DiscreteEnum{TEnum}, DiscreteEnum{TEnum})"]/*'/>
        public static Bernoulli AreEqualAverageConditional(DiscreteEnum<TEnum> A, DiscreteEnum<TEnum> B)
        {
            return DiscreteAreEqualOp.AreEqualAverageConditional(A.GetInternalDiscrete(), B.GetInternalDiscrete());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumAreEqualOp{TEnum}"]/message_doc[@name="AreEqualAverageConditional(TEnum, DiscreteEnum{TEnum})"]/*'/>
        public static Bernoulli AreEqualAverageConditional(TEnum A, DiscreteEnum<TEnum> B)
        {
            return DiscreteAreEqualOp.AreEqualAverageConditional(ToInt(A), B.GetInternalDiscrete());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumAreEqualOp{TEnum}"]/message_doc[@name="AreEqualAverageConditional(DiscreteEnum{TEnum}, TEnum)"]/*'/>
        public static Bernoulli AreEqualAverageConditional(DiscreteEnum<TEnum> A, TEnum B)
        {
            return DiscreteAreEqualOp.AreEqualAverageConditional(A.GetInternalDiscrete(), ToInt(B));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumAreEqualOp{TEnum}"]/message_doc[@name="AreEqualAverageLogarithm(DiscreteEnum{TEnum}, DiscreteEnum{TEnum})"]/*'/>
        public static Bernoulli AreEqualAverageLogarithm(DiscreteEnum<TEnum> A, DiscreteEnum<TEnum> B)
        {
            return DiscreteAreEqualOp.AreEqualAverageLogarithm(A.GetInternalDiscrete(), B.GetInternalDiscrete());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumAreEqualOp{TEnum}"]/message_doc[@name="AreEqualAverageLogarithm(TEnum, DiscreteEnum{TEnum})"]/*'/>
        public static Bernoulli AreEqualAverageLogarithm(TEnum A, DiscreteEnum<TEnum> B)
        {
            return DiscreteAreEqualOp.AreEqualAverageLogarithm(ToInt(A), B.GetInternalDiscrete());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumAreEqualOp{TEnum}"]/message_doc[@name="AreEqualAverageLogarithm(DiscreteEnum{TEnum}, TEnum)"]/*'/>
        public static Bernoulli AreEqualAverageLogarithm(DiscreteEnum<TEnum> A, TEnum B)
        {
            return DiscreteAreEqualOp.AreEqualAverageLogarithm(A.GetInternalDiscrete(), ToInt(B));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumAreEqualOp{TEnum}"]/message_doc[@name="LogAverageFactor(Bernoulli, TEnum, TEnum)"]/*'/>
        public static double LogAverageFactor(Bernoulli areEqual, TEnum A, TEnum B)
        {
            return DiscreteAreEqualOp.LogAverageFactor(areEqual, ToInt(A), ToInt(B));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumAreEqualOp{TEnum}"]/message_doc[@name="LogAverageFactor(bool, TEnum, TEnum)"]/*'/>
        public static double LogAverageFactor(bool areEqual, TEnum A, TEnum B)
        {
            return DiscreteAreEqualOp.LogAverageFactor(areEqual, ToInt(A), ToInt(B));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumAreEqualOp{TEnum}"]/message_doc[@name="LogAverageFactor(bool, TEnum, DiscreteEnum{TEnum})"]/*'/>
        public static double LogAverageFactor(bool areEqual, TEnum A, DiscreteEnum<TEnum> B)
        {
            return DiscreteAreEqualOp.LogAverageFactor(areEqual, ToInt(A), B.GetInternalDiscrete());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumAreEqualOp{TEnum}"]/message_doc[@name="LogAverageFactor(bool, DiscreteEnum{TEnum}, TEnum)"]/*'/>
        public static double LogAverageFactor(bool areEqual, DiscreteEnum<TEnum> A, TEnum B)
        {
            return DiscreteAreEqualOp.LogAverageFactor(areEqual, A.GetInternalDiscrete(), ToInt(B));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumAreEqualOp{TEnum}"]/message_doc[@name="LogEvidenceRatio(bool, DiscreteEnum{TEnum}, DiscreteEnum{TEnum}, DiscreteEnum{TEnum})"]/*'/>
        public static double LogEvidenceRatio(bool areEqual, DiscreteEnum<TEnum> A, DiscreteEnum<TEnum> B, [Fresh] DiscreteEnum<TEnum> to_A)
        {
            return DiscreteAreEqualOp.LogEvidenceRatio(areEqual, A.GetInternalDiscrete(), B.GetInternalDiscrete(), to_A.GetInternalDiscrete());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumAreEqualOp{TEnum}"]/message_doc[@name="LogEvidenceRatio(bool, TEnum, DiscreteEnum{TEnum}, DiscreteEnum{TEnum})"]/*'/>
        public static double LogEvidenceRatio(bool areEqual, TEnum A, DiscreteEnum<TEnum> B, [Fresh] DiscreteEnum<TEnum> to_B)
        {
            return DiscreteAreEqualOp.LogEvidenceRatio(areEqual, ToInt(A), B.GetInternalDiscrete(), to_B.GetInternalDiscrete());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumAreEqualOp{TEnum}"]/message_doc[@name="LogEvidenceRatio(bool, DiscreteEnum{TEnum}, TEnum, DiscreteEnum{TEnum})"]/*'/>
        public static double LogEvidenceRatio(bool areEqual, DiscreteEnum<TEnum> A, TEnum B, [Fresh] DiscreteEnum<TEnum> to_A)
        {
            return DiscreteAreEqualOp.LogEvidenceRatio(areEqual, A.GetInternalDiscrete(), ToInt(B), to_A.GetInternalDiscrete());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumAreEqualOp{TEnum}"]/message_doc[@name="LogEvidenceRatio(bool, TEnum, TEnum)"]/*'/>
        public static double LogEvidenceRatio(bool areEqual, TEnum A, TEnum B)
        {
            return DiscreteAreEqualOp.LogEvidenceRatio(areEqual, ToInt(A), ToInt(B));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteEnumAreEqualOp{TEnum}"]/message_doc[@name="LogEvidenceRatio(Bernoulli)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Bernoulli areEqual)
        {
            return 0.0;
        }

        /// <summary>
        /// Evidence message for VMP.
        /// </summary>
        /// <returns>Zero</returns>
        /// <remarks><para>
        /// In Variational Message Passing, the evidence contribution of a deterministic factor is zero.
        /// </para></remarks>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }
    }
}
