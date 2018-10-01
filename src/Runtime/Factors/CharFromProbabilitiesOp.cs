// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CharFromProbabilitiesOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Char")]
    [Quality(QualityBand.Experimental)]
    public static class CharFromProbabilitiesOp
    {
        #region EP messages

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CharFromProbabilitiesOp"]/message_doc[@name="CharacterAverageConditional(Dirichlet)"]/*'/>
        public static DiscreteChar CharacterAverageConditional([SkipIfUniform] Dirichlet probabilities)
        {
            Discrete resultAsDiscrete = Discrete.Uniform(probabilities.Dimension, probabilities.Sparsity);
            DiscreteFromDirichletOp.SampleAverageConditional(probabilities, resultAsDiscrete);
            return DiscreteChar.FromVector(resultAsDiscrete.GetProbs());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CharFromProbabilitiesOp"]/message_doc[@name="ProbabilitiesAverageConditional(DiscreteChar, Dirichlet, Dirichlet)"]/*'/>
        public static Dirichlet ProbabilitiesAverageConditional([SkipIfUniform] DiscreteChar character, Dirichlet probabilities, Dirichlet result)
        {
            return DiscreteFromDirichletOp.ProbsAverageConditional(new Discrete(character.GetProbs()), probabilities, result);
        }

        #endregion

        #region Evidence messages

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CharFromProbabilitiesOp"]/message_doc[@name="LogEvidenceRatio(Dirichlet, DiscreteChar)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Dirichlet probabilities, DiscreteChar character)
        {
            return 0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CharFromProbabilitiesOp"]/message_doc[@name="LogEvidenceRatio(Dirichlet, Char)"]/*'/>
        public static double LogEvidenceRatio(Dirichlet probabilities, char character)
        {
            return DiscreteFromDirichletOp.LogEvidenceRatio(character, probabilities);
        }

        #endregion
    }
}
