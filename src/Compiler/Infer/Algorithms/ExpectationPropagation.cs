// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.Transforms;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace Microsoft.ML.Probabilistic.Algorithms
{
    /// <summary>
    /// The expectation propagation inference algorithm, see also  
    /// http://research.microsoft.com/~minka/papers/ep/roadmap.html.
    /// </summary>
    public class ExpectationPropagation : AlgorithmBase, IAlgorithm
    {
        #region IAlgorithm Members

        public override Delegate GetVariableFactor(bool derived, bool initialised)
        {
            if (derived)
            {
                if (initialised) return new FuncOut<PlaceHolder, PlaceHolder, PlaceHolder, PlaceHolder>(Clone.DerivedVariableInit);
                else return new FuncOut<PlaceHolder, PlaceHolder, PlaceHolder>(Clone.DerivedVariable);
            }
            else
            {
                if (initialised) return new FuncOut<PlaceHolder, PlaceHolder, PlaceHolder, PlaceHolder>(Clone.VariableInit);
                else return new FuncOut<PlaceHolder, PlaceHolder, PlaceHolder>(Clone.Variable);
            }
        }

        /// <summary>
        /// Gets the suffix for Expectation Propagation operator methods
        /// </summary>
        /// <param name="factorAttributes"></param>
        /// <returns></returns>
        public override string GetOperatorMethodSuffix(List<ICompilerAttribute> factorAttributes)
        {
            return "AverageConditional";
        }

        /// <summary>
        /// Gets the suffix for Expectation Propagation evidence method
        /// </summary>
        /// <param name="factorAttributes"></param>
        /// <returns></returns>
        public override string GetEvidenceMethodName(List<ICompilerAttribute> factorAttributes)
        {
            // If this is changed, must also change #define at top of GateEnter.cs and GateExit.cs
            return "LogEvidenceRatio";
        }

        /// <summary>
        /// Name of the algorithm
        /// </summary>
        public override string Name
        {
            get { return "ExpectationPropagation"; }
        }

        /// <summary>
        /// Short name of the algorithm
        /// </summary>
        public override string ShortName
        {
            get { return "EP"; }
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
            if (alg2 is VariationalMessagePassing)
            {
                args.Add(-1.0);
                return new MethodReference(typeof(ShiftAlpha), isFromFactor ? "FromFactor<>" : "ToFactor<>")
                {
                    TypeArguments = new Type[] { channelType }
                };
            }
            throw new InferCompilerException("Cannot convert from " + Name + " to " + alg2.Name);
        }

        #endregion
    }
}