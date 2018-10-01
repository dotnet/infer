// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// A condition expression
    /// </summary>
    internal class XConditionExpression : XExpression, IConditionExpression
    {
        #region Fields

        private IExpression condition;
        private IExpression thenExpr;
        private IExpression elseExpr;

        #endregion

        #region IConditionExpression Members

        /// <summary>
        /// The condition
        /// </summary>
        public IExpression Condition
        {
            get { return this.condition; }
            set { this.condition = value; }
        }

        /// <summary>
        /// Else expression
        /// </summary>
        public IExpression Else
        {
            get { return this.elseExpr; }
            set { this.elseExpr = value; }
        }

        /// <summary>
        /// Then expression
        /// </summary>
        public IExpression Then
        {
            get { return this.thenExpr; }
            set { this.thenExpr = value; }
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

            IConditionExpression expression = obj as IConditionExpression;
            if (expression == null)
                return false;

            return
                this.Condition.Equals(expression.Condition) &&
                this.Then.Equals(expression.Then) &&
                this.Else.Equals(expression.Else);
        }

        public override int GetHashCode()
        {
            return this.Condition.GetHashCode();
        }

        #endregion Object Overrides

        /// <summary>
        /// Get expression type
        /// </summary>
        public override Type GetExpressionType()
        {
            return null;
        }
    }
}