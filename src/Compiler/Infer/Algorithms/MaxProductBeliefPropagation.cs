// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.Transforms;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace Microsoft.ML.Probabilistic.Algorithms
{
    /// <summary>
    /// Max product belief propagation.
    /// </summary>
    public class MaxProductBeliefPropagation : AlgorithmBase, IAlgorithm
    {
        #region IAlgorithm Members

        public override Delegate GetVariableFactor(bool derived, bool initialised)
        {
            return new FuncOut<PlaceHolder, PlaceHolder, PlaceHolder>(Clone.VariableMax);
        }

        /// <summary>
        /// Gets the suffix for Max Product operator methods
        /// </summary>
        /// <param name="factorAttributes"></param>
        /// <returns></returns>
        public override string GetOperatorMethodSuffix(List<ICompilerAttribute> factorAttributes)
        {
            return "MaxConditional";
        }

        /// <summary>
        /// Gets the suffix for Max Product evidence method
        /// </summary>
        /// <param name="factorAttributes"></param>
        /// <returns></returns>
        public override string GetEvidenceMethodName(List<ICompilerAttribute> factorAttributes)
        {
            return "NotYetSupported";
        }

        /// <summary>
        /// Name of the algorithm
        /// </summary>
        public override string Name
        {
            get { return "MaxProductBP"; }
        }

        /// <summary>
        /// Short name of the algorithm
        /// </summary>
        public override string ShortName
        {
            get { return "MaxProd"; }
        }

        /// <summary>
        /// Gets the operator which converts a message to/from another algorithm
        /// </summary>
        /// <param name="channelType">Type of message</param>
        /// <param name="alg2">The other algorithm</param>
        /// <param name="isFromFactor">True if from, false if to</param>
        /// <param name="args">Where to add arguments of the operator</param>
        /// <returns>A method reference for the operator</returns>
        public override MethodReference GetAlgorithmConversionOperator(Type channelType, IAlgorithm alg2, bool isFromFactor, List<object> args)
        {
            throw new InferCompilerException("Cannot convert from " + Name + " to " + alg2.Name);
        }

        public override IExpression GetMessagePrototype(ChannelInfo channelInfo, MessageDirection direction, IExpression marginalPrototypeExpression, string path,
                                                        IList<QueryType> queryTypes)
        {
            if (marginalPrototypeExpression.GetExpressionType() == typeof (Discrete))
            {
                return CodeBuilder.Instance.StaticMethod(new Func<Discrete, UnnormalizedDiscrete>(UnnormalizedDiscrete.FromDiscrete), marginalPrototypeExpression);
            }
            return base.GetMessagePrototype(channelInfo, direction, marginalPrototypeExpression, path, queryTypes);
        }

        #endregion
    }
}