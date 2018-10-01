// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// A checked arithmetic expression
    /// </summary>
    internal class XCheckedExpression : XExpression, ICheckedExpression
    {
        #region Fields

        private IExpression expression;

        #endregion

        #region Constructor

        public XCheckedExpression()
        {
        }

        #endregion

        #region ICheckedExpression Members

        /// <summary>
        /// The expression
        /// </summary>
        public IExpression Expression
        {
            get { return this.expression; }
            set { this.expression = value; }
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

            ICheckedExpression expression = obj as ICheckedExpression;
            if (expression == null)
                return false;

            return this.Expression.Equals(expression.Expression);
        }

        public override int GetHashCode()
        {
            return this.Expression.GetHashCode();
        }

        #endregion Object Overrides

        /// <summary>
        /// Get expression type
        /// </summary>
        public override Type GetExpressionType()
        {
            return this.Expression.GetExpressionType();
        }
    }
}
