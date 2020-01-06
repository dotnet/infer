// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Diagnostics;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaProductOp_Laplace"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
    [Buffers("Q")]
    [Quality(QualityBand.Experimental)]
    public static class GammaProductOp_Laplace
    {
        public static bool ForceProper = true;

        // derivatives of the factor marginalized over Product and A
        private static double[] dlogfs(double b, Gamma product, Gamma A)
        {
            return GammaPowerProductOp_Laplace.dlogfs(b, GammaPower.FromGamma(product, 1), GammaPower.FromGamma(A, 1));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaProductOp_Laplace"]/message_doc[@name="QInit()"]/*'/>
        [Skip]
        public static Gamma QInit()
        {
            return Gamma.Uniform();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaProductOp_Laplace"]/message_doc[@name="Q(Gamma, Gamma, Gamma)"]/*'/>
        [Fresh]
        public static Gamma Q(Gamma product, [Proper] Gamma A, [Proper] Gamma B)
        {
            return GammaPowerProductOp_Laplace.Q(GammaPower.FromGamma(product, 1), GammaPower.FromGamma(A, 1), GammaPower.FromGamma(B, 1));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaProductOp_Laplace"]/message_doc[@name="LogAverageFactor(Gamma, Gamma, Gamma, Gamma)"]/*'/>
        public static double LogAverageFactor(Gamma product, Gamma A, Gamma B, Gamma q)
        {
            return GammaPowerProductOp_Laplace.LogAverageFactor(GammaPower.FromGamma(product, 1), GammaPower.FromGamma(A, 1), GammaPower.FromGamma(B, 1), q);
        }

        private static double LogAverageFactor_slow(Gamma product, Gamma A, [Proper] Gamma B)
        {
            return LogAverageFactor(product, A, B, Q(product, A, B));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaProductOp_Laplace"]/message_doc[@name="LogEvidenceRatio(Gamma, Gamma, Gamma, Gamma, Gamma)"]/*'/>
        public static double LogEvidenceRatio([SkipIfUniform] Gamma product, Gamma A, Gamma B, Gamma to_product, Gamma q)
        {
            return LogAverageFactor(product, A, B, q) - to_product.GetLogAverageOf(product);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaProductOp_Laplace"]/message_doc[@name="LogEvidenceRatio(double, Gamma, Gamma, Gamma)"]/*'/>
        public static double LogEvidenceRatio(double product, Gamma A, Gamma B, Gamma q)
        {
            return LogAverageFactor(Gamma.PointMass(product), A, B, q);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaProductOp_Laplace"]/message_doc[@name="BAverageConditional(Gamma, Gamma, Gamma, Gamma)"]/*'/>
        public static Gamma BAverageConditional([SkipIfUniform] Gamma product, [Proper] Gamma A, [Proper] Gamma B, Gamma q)
        {
            return ToGamma(GammaPowerProductOp_Laplace.BAverageConditional(GammaPower.FromGamma(product, 1), GammaPower.FromGamma(A, 1), GammaPower.FromGamma(B, 1), q, GammaPower.Uniform(1)));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaProductOp_Laplace"]/message_doc[@name="ProductAverageConditional(Gamma, Gamma, Gamma, Gamma)"]/*'/>
        public static Gamma ProductAverageConditional(Gamma product, [Proper] Gamma A, [SkipIfUniform] Gamma B, Gamma q)
        {
            return ToGamma(GammaPowerProductOp_Laplace.ProductAverageConditional(GammaPower.FromGamma(product, 1), GammaPower.FromGamma(A, 1), GammaPower.FromGamma(B, 1), q, GammaPower.Uniform(1)));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaProductOp_Laplace"]/message_doc[@name="AAverageConditional(Gamma, Gamma, Gamma, Gamma)"]/*'/>
        public static Gamma AAverageConditional([SkipIfUniform] Gamma product, Gamma A, [SkipIfUniform] Gamma B, Gamma q)
        {
            return ToGamma(GammaPowerProductOp_Laplace.AAverageConditional(GammaPower.FromGamma(product, 1), GammaPower.FromGamma(A, 1), GammaPower.FromGamma(B, 1), q, GammaPower.Uniform(1)));
        }

        public static Gamma ToGamma(GammaPower gammaPower)
        {
            if (gammaPower.Power != 1) throw new Exception("gammaPower.Power != 1");
            return Gamma.FromShapeAndRate(gammaPower.Shape, gammaPower.Rate);
        }
    }

}
