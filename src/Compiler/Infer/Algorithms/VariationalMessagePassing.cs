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
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// The variational message passing algorithm, see also  
    /// http://www.johnwinn.org/Research/VMP.html and 
    /// http://en.wikipedia.org/wiki/Variational_message_passing.
    /// </summary>
    public class VariationalMessagePassing : AlgorithmBase, IAlgorithm
    {
        internal class GateExitRandomVariable : ICompilerAttribute
        {
        }

        public bool UseGateExitRandom;
        public bool UseDerivMessages;

        #region IAlgorithm Members

        public override Delegate GetVariableFactor(bool derived, bool initialised)
        {
            if (derived)
            {
                if (initialised) return new FuncOut<PlaceHolder, PlaceHolder, PlaceHolder, PlaceHolder>(Clone.DerivedVariableInitVmp);
                else return new FuncOut<PlaceHolder, PlaceHolder, PlaceHolder>(Clone.DerivedVariableVmp);
            }
            else
            {
                if (initialised) return new FuncOut<PlaceHolder, PlaceHolder, PlaceHolder, PlaceHolder>(Clone.VariableInit);
                else return new FuncOut<PlaceHolder, PlaceHolder, PlaceHolder>(Clone.Variable);
            }
        }

        /// <summary>
        /// Gets the suffix for variational message passing operator methods
        /// </summary>
        /// <param name="factorAttributes"></param>
        /// <returns></returns>
        public override string GetOperatorMethodSuffix(List<ICompilerAttribute> factorAttributes)
        {
            return "AverageLogarithm";
        }

        /// <summary>
        /// Gets the suffix for variational message passing evidence method
        /// </summary>
        /// <param name="factorAttributes"></param>
        /// <returns></returns>
        public override string GetEvidenceMethodName(List<ICompilerAttribute> factorAttributes)
        {
            return "AverageLogFactor";
        }

        /// <summary>
        /// Name of the algorithm
        /// </summary>
        public override string Name
        {
            get { return "VariationalMessagePassing"; }
        }

        /// <summary>
        /// Short name of the algorithm
        /// </summary>
        public override string ShortName
        {
            get { return "VMP"; }
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
            if (alg2 is ExpectationPropagation)
            {
                args.Add(1.0);
                return new MethodReference(typeof(ShiftAlpha), isFromFactor ? "FromFactor<>" : "ToFactor<>")
                {
                    TypeArguments = new Type[] { channelType }
                };
            }
            throw new InferCompilerException("Cannot convert from " + Name + " to " + alg2.Name);
        }

        #endregion
    }

    /*public class VmpDeterministic : IAlgorithm
    {
        public string GetOperatorMethodSuffix(List<object> factorAttributes)
        {
            return "VmpDeterministic";
        }

        public MethodReference GetAlgorithmConversionOperator(Type channelType, IAlgorithm alg2, bool isFromFactor, List<object> args)
        {
            if (alg2 is VariationalMessagePassing)
            {
                return null;
            }
        }

        public string Name
        {
            get { return "VariationalMessagePassing(Deterministic)"; }
        }
    }*/
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}