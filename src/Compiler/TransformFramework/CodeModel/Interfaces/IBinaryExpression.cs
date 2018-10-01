// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Binary operators
    /// </summary>
    public enum BinaryOperator
    {
        /// <summary>
        /// Add operator
        /// </summary>
        Add = 0,

        /// <summary>
        /// Subtraction operator
        /// </summary>
        Subtract = 1,

        /// <summary>
        /// Multiplication operator
        /// </summary>
        Multiply = 2,

        /// <summary>
        /// Division operator
        /// </summary>
        Divide = 3,

        /// <summary>
        /// Modulus operator
        /// </summary>
        Modulus = 4,

        /// <summary>
        /// Shift left operator
        /// </summary>
        ShiftLeft = 5,

        /// <summary>
        /// Shift right operator
        /// </summary>
        ShiftRight = 6,

        /// <summary>
        /// Identity equality operator
        /// </summary>
        IdentityEquality = 7,

        /// <summary>
        /// Identity inequality operator
        /// </summary>
        IdentityInequality = 8,

        /// <summary>
        /// Equality operator
        /// </summary>
        ValueEquality = 9,

        /// <summary>
        /// Inequality operator
        /// </summary>
        ValueInequality = 10,

        /// <summary>
        /// Bitwise OR operator
        /// </summary>
        BitwiseOr = 11,

        /// <summary>
        /// Bitwise AND operator
        /// </summary>
        BitwiseAnd = 12,

        /// <summary>
        /// Bitwise Exclusive OR operator
        /// </summary>
        BitwiseExclusiveOr = 13,

        /// <summary>
        /// Boolean OR operator
        /// </summary>
        BooleanOr = 14,

        /// <summary>
        /// Boolean AND operator
        /// </summary>
        BooleanAnd = 15,

        /// <summary>
        /// Less than operator
        /// </summary>
        LessThan = 16,

        /// <summary>
        /// Less than or equal operator
        /// </summary>
        LessThanOrEqual = 17,

        /// <summary>
        /// Greater than operator
        /// </summary>
        GreaterThan = 18,

        /// <summary>
        /// Greater than or equal operator
        /// </summary>
        GreaterThanOrEqual = 19,
    }

    /// <summary>
    /// Binary expression
    /// </summary>
    public interface IBinaryExpression : IExpression
    {
        /// <summary>
        /// Expression to the left of the operator
        /// </summary>
        IExpression Left { get; set; }

        /// <summary>
        /// The operator
        /// </summary>
        BinaryOperator Operator { get; set; }

        /// <summary>
        /// Expression to the right of the operator
        /// </summary>
        IExpression Right { get; set; }
    }
}