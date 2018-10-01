// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.Transforms;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Compiler;

namespace Microsoft.ML.Probabilistic.Models.Attributes
{
    /// <summary>
    /// Interface for inference algorithms
    /// </summary>
    public interface IAlgorithm
    {
        /// <summary>
        /// The name
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Short name for the inference algorithm
        /// </summary>
        string ShortName { get; }

        Delegate GetVariableFactor(bool derived, bool initialised);

        /// <summary>
        /// Gets the suffix for this algorithm's operator methods
        /// </summary>
        /// <param name="factorAttributes"></param>
        /// <returns></returns>
        string GetOperatorMethodSuffix(List<ICompilerAttribute> factorAttributes);

        /// <summary>
        /// Gets the operator which converts a message to/from another algorithm
        /// </summary>
        /// <param name="channelType">Type of message</param>
        /// <param name="alg2">The other algorithm</param>
        /// <param name="isFromFactor">True if from, false if to</param>
        /// <param name="args">Where to add arguments of the operator</param>
        /// <returns>A method reference for the operator</returns>
        MethodReference GetAlgorithmConversionOperator(Type channelType, IAlgorithm alg2, bool isFromFactor, List<object> args);

        /// <summary>
        /// Gets the suffix for this algorithm's evidence method
        /// </summary>
        /// <param name="factorAttributes"></param>
        /// <returns></returns>
        string GetEvidenceMethodName(List<ICompilerAttribute> factorAttributes);

        /// <summary>
        /// Get the message prototype for this algorithm in the specified direction
        /// </summary>
        /// <param name="channelInfo">The channel information</param>
        /// <param name="direction">The direction</param>
        /// <param name="marginalPrototypeExpression">The marginal prototype expression</param>
        /// <param name="path">Sub-channel path</param>
        /// <param name="queryTypes"></param>
        /// <returns></returns>
        IExpression GetMessagePrototype(
            ChannelInfo channelInfo,
            MessageDirection direction,
            IExpression marginalPrototypeExpression,
            string path,
            IList<QueryType> queryTypes);

        /// <summary>
        /// Allows the algorithm to modify the attributes on a factor. For example context-specific
        /// message attributes on a method invoke expression
        /// </summary>
        /// <param name="factorExpression">The expression</param>
        /// <param name="factorAttributes">Attribute registry</param>
        /// <returns></returns>
        void ModifyFactorAttributes(IExpression factorExpression, AttributeRegistry<object, ICompilerAttribute> factorAttributes);

        /// <summary>
        /// Get the default inference query types for a variable for this algorithm.
        /// </summary>
        void ForEachDefaultQueryType(Action<QueryType> action);

        /// <summary>
        /// Get the query type binding - this is the path to the given query type
        /// relative to the raw marginal type.
        /// </summary>
        /// <param name="qt">The query type</param>
        /// <returns></returns>
        string GetQueryTypeBinding(QueryType qt);

        /// <summary>
        /// Default number of iterations for this algorithm
        /// </summary>
        int DefaultNumberOfIterations { get; set; }
    }
}