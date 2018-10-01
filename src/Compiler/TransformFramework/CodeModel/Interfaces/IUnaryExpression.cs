// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Unary operators
    /// </summary>
    public enum UnaryOperator
    {
        /// <summary>
        /// Negation operator
        /// </summary>
        Negate = 0,

        /// <summary>
        /// Boolean NOT operator
        /// </summary>
        BooleanNot = 1,

        /// <summary>
        /// Bitwise NOT operator
        /// </summary>
        BitwiseNot = 2,

        /// <summary>
        /// Pre-increment operator
        /// </summary>
        PreIncrement = 3,

        /// <summary>
        /// Pre-decrement operator
        /// </summary>
        PreDecrement = 4,

        /// <summary>
        /// Post-increment operator
        /// </summary>
        PostIncrement = 5,

        /// <summary>
        /// Post-decrement operator
        /// </summary>
        PostDecrement = 6,
    }

    /// <summary>
    /// Unary expression
    /// </summary>
    public interface IUnaryExpression : IExpression
    {
        /// <summary>
        /// The source expression
        /// </summary>
        IExpression Expression { get; set; }

        /// <summary>
        /// The operator that acts on the source expression
        /// </summary>
        UnaryOperator Operator { get; set; }
    }
}