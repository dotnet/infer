// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Unary expression
    /// </summary>
    internal class XUnaryExpression : XExpression, IUnaryExpression
    {
        #region Fields

        private UnaryOperator unaryOp;
        private IExpression expression;

        #endregion

        #region IUnaryExpression Members

        /// <summary>
        /// The expression operated on
        /// </summary>
        public IExpression Expression
        {
            get { return this.expression; }
            set { this.expression = value; }
        }

        /// <summary>
        /// The operator
        /// </summary>
        public UnaryOperator Operator
        {
            get { return this.unaryOp; }
            set { this.unaryOp = value; }
        }

        #endregion

        #region Object Overrides

        public override string ToString()
        {
            // Base class will do the right thing
            return base.ToString();
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            IUnaryExpression expr = obj as IUnaryExpression;
            if (expr == null)
                return false;

            return Expression.Equals(expr.Expression) && Operator.Equals(expr.Operator);
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
            return Expression.GetExpressionType();
        }
    }
}