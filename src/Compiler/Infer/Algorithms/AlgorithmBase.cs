// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.Transforms;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace Microsoft.ML.Probabilistic.Algorithms
{
    /// <summary>
    /// Abstract base class for all algorithms
    /// </summary>
    public abstract class AlgorithmBase : IAlgorithm
    {
        /// <summary>
        /// The algorithm's name
        /// </summary>
        public abstract string Name { get; }

        /// <summary>
        /// Short name for the inference algorithm
        /// </summary>
        public abstract string ShortName { get; }

        public abstract Delegate GetVariableFactor(bool derived, bool initialised);

        /// <summary>
        /// Algorithm's operator suffix - used in in message update methods
        /// </summary>
        /// <param name="factorAttributes"></param>
        /// <returns></returns>
        public abstract string GetOperatorMethodSuffix(List<ICompilerAttribute> factorAttributes);

        /// <summary>
        /// Gets the operator which converts a message to/from another algorithm
        /// </summary>
        /// <param name="channelType">Type of message</param>
        /// <param name="alg2">The other algorithm</param>
        /// <param name="isFromFactor">True if from, false if to</param>
        /// <param name="args">Where to add arguments of the operator</param>
        /// <returns>A method reference for the operator</returns>
        public abstract MethodReference GetAlgorithmConversionOperator(Type channelType, IAlgorithm alg2, bool isFromFactor, List<object> args);

        /// <summary>
        /// Gets the suffix for this algorithm's evidence method
        /// </summary>
        /// <param name="factorAttributes"></param>
        /// <returns></returns>
        public abstract string GetEvidenceMethodName(List<ICompilerAttribute> factorAttributes);

        /// <summary>
        /// Get the message prototype for this algorithm in the specified direction
        /// </summary>
        /// <param name="channelInfo">The channel information</param>
        /// <param name="direction">The direction</param>
        /// <param name="marginalPrototypeExpression">The marginal prototype expression</param>
        /// <param name="path">Path name of message</param>
        /// <param name="queryTypes"></param>
        /// <returns></returns>
        public virtual IExpression GetMessagePrototype(
            ChannelInfo channelInfo, MessageDirection direction, IExpression marginalPrototypeExpression, string path, IList<QueryType> queryTypes)
        {
            return marginalPrototypeExpression;
        }

        /// <summary>
        /// Allows the algorithm to modify the attributes on a factor. For example context-specific
        /// message attributes on a method invoke expression
        /// </summary>
        /// <param name="factorExpression">The expression</param>
        /// <param name="factorAttributes">Attribute registry</param>
        /// <returns></returns>
        public virtual void ModifyFactorAttributes(IExpression factorExpression, AttributeRegistry<object, ICompilerAttribute> factorAttributes)
        {
            // By default, remove any message path attributes
            factorAttributes.Remove<MessagePathAttribute>(factorExpression);
            return;
        }

        /// <summary>
        /// Get the default inference query types for a variable for this algorithm.
        /// </summary>
        public virtual void ForEachDefaultQueryType(Action<QueryType> action)
        {
            action(QueryTypes.Marginal);
        }

        /// <summary>
        /// Get the query type binding - this is the path to the given query type
        /// relative to the raw marginal type.
        /// </summary>
        /// <param name="qt">The query type</param>
        /// <returns></returns>
        public virtual string GetQueryTypeBinding(QueryType qt)
        {
            return "";
        }

        private int defaultNumberOfIterations = -1;

        /// <summary>
        /// Default number of iterations for this algorithm
        /// </summary>
        public virtual int DefaultNumberOfIterations
        {
            get { return (defaultNumberOfIterations < 0) ? 50 : defaultNumberOfIterations; }
            set { defaultNumberOfIterations = value; }
        }
    }
}