// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors.Attributes;

namespace Microsoft.ML.Probabilistic.Factors
{
    /// <summary>
    /// UsesEqualDef operator to combine Gaussian and nonconjugate messages. 
    /// </summary>
    [Quality(QualityBand.Experimental)]
    public static class NonconjugateUsesEqualDefOp
    {
        /// <summary>
        /// VMP Messages to Uses
        /// </summary>
        /// <param name="Uses">Nonconjugate messages from uses</param>
        /// <param name="Def">Gaussian message from Definition</param>
        /// <param name="result">Previous message</param>
        /// <returns></returns>
        public static DistributionStructArray<Gaussian, double> UsesAverageLogarithm(DistributionStructArray<NonconjugateGaussian, double>[] Uses,
                                                                                     DistributionStructArray<Gaussian, double> Def,
                                                                                     DistributionStructArray<Gaussian, double> result)
        {
            // multiply the messages together
            DistributionStructArray<NonconjugateGaussian, double> prod = Uses[0];
            for (int i = 1; i < Uses.Length; i++)
                prod.SetToProduct(prod, Uses[i]);
            // convert to Gaussians
            for (int i = 0; i < prod.Count; i++)
            {
                result[i] = prod[i].GetGaussian(true);
                if ((!result[i].IsProper() && !result[i].IsUniform()) || double.IsNaN(result[i].Precision))
                    throw new InferRuntimeException("UsesAverageLogarithm ");
            }
            result.SetToProduct(result, Def);
            return result;
        }

        /// <summary>
        /// The marginal distribution is the same as the message to uses.  
        /// </summary>
        /// <param name="Uses"></param>
        /// <param name="Def"></param>
        /// <param name="result"></param>
        /// <returns></returns>
        public static DistributionStructArray<Gaussian, double> MarginalAverageLogarithm(DistributionStructArray<NonconjugateGaussian, double>[] Uses,
                                                                                         DistributionStructArray<Gaussian, double> Def,
                                                                                         DistributionStructArray<Gaussian, double> result)
        {
            return UsesAverageLogarithm(Uses, Def, result);
        }


        public static Gaussian UsesAverageLogarithm(NonconjugateGaussian[] Uses, Gaussian Def, Gaussian result)
        {
            NonconjugateGaussian prod = Uses[0];
            for (int i = 1; i < Uses.Length; i++)
                prod.SetToProduct(prod, Uses[i]);
            result = prod.GetGaussian(true);
            result.SetToProduct(result, Def);
            return result;
        }

        public static Gaussian MarginalAverageLogarithm(NonconjugateGaussian[] Uses, Gaussian Def, Gaussian result)
        {
            return UsesAverageLogarithm(Uses, Def, result);
        }
    }
}
