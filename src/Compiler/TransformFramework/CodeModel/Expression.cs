// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Base class for all expressions
    /// </summary>
    public abstract class XExpression : IExpression
    {
        /// <summary>
        /// Constructor
        /// </summary>
        protected XExpression()
        {
        }

        /// <summary>
        /// Get the expression type
        /// </summary>
        /// <returns></returns>
        public abstract Type GetExpressionType();

        /// <summary>
        /// ToString override
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            ILanguageWriter writer = new CSharpWriter() as ILanguageWriter;
            return writer.ExpressionSource(this);
        }
    }
}