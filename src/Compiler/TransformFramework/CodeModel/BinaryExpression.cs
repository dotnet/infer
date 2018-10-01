// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// A bibary expression
    /// </summary>
    internal class XBinaryExpression : XExpression, IBinaryExpression
    {
        #region Fields

        private BinaryOperator oper;
        private IExpression left;
        private IExpression right;

        #endregion

        #region IBinaryExpression Members

        /// <summary>
        /// The expression to the left of the operator
        /// </summary>
        public IExpression Left
        {
            get { return this.left; }
            set { this.left = value; }
        }

        /// <summary>
        /// The operator
        /// </summary>
        public BinaryOperator Operator
        {
            get { return this.oper; }
            set { this.oper = value; }
        }

        /// <summary>
        /// The expression to the right of the operator
        /// </summary>
        public IExpression Right
        {
            get { return this.right; }
            set { this.right = value; }
        }

        #endregion

        #region Object Overrides

        public override string ToString()
        {
            // Base expression will deal with this correctly
            return base.ToString();
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            IBinaryExpression expression = obj as IBinaryExpression;
            if (expression == null)
                return false;

            return
                this.Left.Equals(expression.Left) &&
                this.Right.Equals(expression.Right) &&
                this.Operator.Equals(expression.Operator);
        }

        public override int GetHashCode()
        {
            return this.Operator.GetHashCode();
        }

        #endregion Object Overrides

        /// <summary>
        /// Get expression type
        /// </summary>
        public override Type GetExpressionType()
        {
            if (Operator == BinaryOperator.ValueEquality ||
                Operator == BinaryOperator.ValueInequality ||
                Operator == BinaryOperator.IdentityEquality ||
                Operator == BinaryOperator.IdentityInequality ||
                Operator == BinaryOperator.BooleanAnd ||
                Operator == BinaryOperator.BooleanOr ||
                Operator == BinaryOperator.GreaterThan ||
                Operator == BinaryOperator.GreaterThanOrEqual ||
                Operator == BinaryOperator.LessThan ||
                Operator == BinaryOperator.LessThanOrEqual)
                return typeof (bool);
            else return left.GetExpressionType();
        }
    }
}