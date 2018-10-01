// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Reference: Dan Huttenlocher - Distance Transform paper

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <summary>
    /// Provides useful factors for undirected models.
    /// </summary>
    public class Undirected
    {
        /// <summary>
        /// Implements an integer Potts potential which has the value of 1 if a=b and exp(-logCost) otherwise.
        /// </summary>
        /// <remarks>
        /// See http://en.wikipedia.org/wiki/Potts_model
        /// </remarks>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="logCost"></param>
        public static void Potts(int a, int b, double logCost)
        {
            throw new NotImplementedException("Deterministic form of Potts not yet implemented");
        }

        /// <summary>
        /// Implements an boolean Potts potential which has the value of 1 if a=b and exp(-logCost) otherwise.
        /// </summary>
        /// <remarks>
        /// See http://en.wikipedia.org/wiki/Potts_model
        /// </remarks>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="logCost"></param>
        public static void Potts(bool a, bool b, double logCost)
        {
            throw new NotImplementedException("Deterministic form of Potts not yet implemented");
        }


        /// <summary>
        /// Implements a linear difference potential which has the value of exp( -|a-b|* logUnitCost ).
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="logUnitCost"></param>
        public static void Linear(int a, int b, double logUnitCost)
        {
            throw new NotImplementedException("Deterministic form of linear difference not yet implemented");
        }

        /// <summary>
        /// Implements a truncated linear difference potential which has the value of  exp( - min( |a-b|* logUnitCost, maxCost) ).
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="logUnitCost"></param>
        /// <param name="maxCost"></param>
        public static void LinearTrunc(int a, int b, double logUnitCost, double maxCost)
        {
            throw new NotImplementedException("Deterministic form of truncated linear difference not yet implemented");
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PottsIntOp"]/doc/*'/>
    [FactorMethod(typeof(Undirected), "Potts", typeof(int), typeof(int), typeof(double))]
    [Quality(QualityBand.Experimental)]
    public class PottsIntOp
    {
        //-- Max product ----------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PottsIntOp"]/message_doc[@name="AMaxConditional(UnnormalizedDiscrete, double, UnnormalizedDiscrete)"]/*'/>
        public static UnnormalizedDiscrete AMaxConditional(UnnormalizedDiscrete B, double logCost, UnnormalizedDiscrete result)
        {
            double max = B.GetWorkspace().Max();
            double[] source = B.GetWorkspace().SourceArray;
            double[] target = result.GetWorkspace().SourceArray;
            for (int i = 0; i < target.Length; i++)
                target[i] = Math.Max(max - logCost, source[i]);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PottsIntOp"]/message_doc[@name="BMaxConditional(UnnormalizedDiscrete, double, UnnormalizedDiscrete)"]/*'/>
        public static UnnormalizedDiscrete BMaxConditional(UnnormalizedDiscrete A, double logCost, UnnormalizedDiscrete result)
        {
            return AMaxConditional(A, logCost, result);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PottsBoolOp"]/doc/*'/>
    [FactorMethod(typeof(Undirected), "Potts", typeof(bool), typeof(bool), typeof(double))]
    [Quality(QualityBand.Experimental)]
    public class PottsBoolOp
    {
        //-- Max product ----------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PottsBoolOp"]/message_doc[@name="AMaxConditional(Bernoulli, double)"]/*'/>
        public static Bernoulli AMaxConditional(Bernoulli B, double logCost)
        {
            return Bernoulli.FromLogOdds(Math.Min(B.LogOdds, logCost));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PottsBoolOp"]/message_doc[@name="BMaxConditional(Bernoulli, double)"]/*'/>
        public static Bernoulli BMaxConditional(Bernoulli A, double logCost)
        {
            return AMaxConditional(A, logCost);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LinearOp"]/doc/*'/>
    [FactorMethod(typeof(Undirected), "Linear")]
    [Quality(QualityBand.Experimental)]
    public class LinearOp
    {
        //-- Max product ----------------------------------------------------------------------
        
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LinearOp"]/message_doc[@name="AMaxConditional(UnnormalizedDiscrete, double, UnnormalizedDiscrete)"]/*'/>
        public static UnnormalizedDiscrete AMaxConditional(UnnormalizedDiscrete B, double logUnitCost, UnnormalizedDiscrete result)
        {
            double[] source = B.GetWorkspace().SourceArray;
            double[] target = result.GetWorkspace().SourceArray;
            // forward pass
            target[0] = source[0];
            for (int i = 1; i < target.Length; i++)
            {
                target[i] = Math.Max(source[i], target[i - 1] - logUnitCost);
            }
            // reverse pass
            for (int i = target.Length - 2; i >= 0; i--)
            {
                target[i] = Math.Max(target[i], target[i + 1] - logUnitCost);
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LinearOp"]/message_doc[@name="BMaxConditional(UnnormalizedDiscrete, double, UnnormalizedDiscrete)"]/*'/>
        public static UnnormalizedDiscrete BMaxConditional(UnnormalizedDiscrete A, double logUnitCost, UnnormalizedDiscrete result)
        {
            return AMaxConditional(A, logUnitCost, result);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LinearTruncOp"]/doc/*'/>
    [FactorMethod(typeof(Undirected), "LinearTrunc")]
    [Quality(QualityBand.Experimental)]
    public class LinearTruncOp
    {
        //-- Max product ----------------------------------------------------------------------
        
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LinearTruncOp"]/message_doc[@name="AMaxConditional(UnnormalizedDiscrete, double, double, UnnormalizedDiscrete)"]/*'/>
        public static UnnormalizedDiscrete AMaxConditional(UnnormalizedDiscrete B, double logUnitCost, double maxCost, UnnormalizedDiscrete result)
        {
            double[] source = B.GetWorkspace().SourceArray;
            double[] target = result.GetWorkspace().SourceArray;
            // forward pass
            target[0] = source[0];
            double max = source[0];
            for (int i = 1; i < target.Length; i++)
            {
                max = Math.Max(max, source[i]);
                target[i] = Math.Max(source[i], target[i - 1] - logUnitCost);
            }
            // reverse pass
            double maxLessCost = max - maxCost;
            target[target.Length - 1] = Math.Max(target[target.Length - 1], maxLessCost);
            for (int i = target.Length - 2; i >= 0; i--)
            {
                target[i] = Math.Max(Math.Max(target[i], target[i + 1] - logUnitCost), maxLessCost);
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LinearTruncOp"]/message_doc[@name="BMaxConditional(UnnormalizedDiscrete, double, double, UnnormalizedDiscrete)"]/*'/>
        public static UnnormalizedDiscrete BMaxConditional(UnnormalizedDiscrete A, double logUnitCost, double maxCost, UnnormalizedDiscrete result)
        {
            return AMaxConditional(A, logUnitCost, maxCost, result);
        }
    }
}
