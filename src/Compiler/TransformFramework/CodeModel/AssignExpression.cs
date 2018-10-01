// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Assign expression
    /// </summary>
    internal class XAssignExpression : XExpression, IAssignExpression
    {
        #region Fields

        private IExpression expression;
        private IExpression target;

        #endregion

        #region IAssignExpression Members

        /// <summary>
        /// The expression
        /// </summary>
        public IExpression Expression
        {
            get { return this.expression; }
            set { this.expression = value; }
        }

        /// <summary>
        /// The target
        /// </summary>
        public IExpression Target
        {
            get { return this.target; }
            set { this.target = value; }
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

            IAssignExpression expression = obj as IAssignExpression;
            if (expression == null)
                return false;

            return
                this.Expression.Equals(expression.Expression) &&
                this.Target.Equals(expression.Target);
        }

        public override int GetHashCode()
        {
            return this.Target.GetHashCode() + this.Expression.GetHashCode();
        }

        #endregion Object Overrides

        /// <summary>
        /// Get expression type
        /// </summary>
        public override Type GetExpressionType()
        {
            return this.target.GetExpressionType();
        }
    }
}