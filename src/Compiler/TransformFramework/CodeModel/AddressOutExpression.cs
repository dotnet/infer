// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Represents an 'out' argument expression
    /// </summary>
    internal class XAddressOutExpression : XExpression, IAddressOutExpression
    {
        #region Fields

        private IExpression expression;

        #endregion

        #region IAddressOutExpression Members

        /// <summary>
        /// Expression
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

            IAddressOutExpression expression = obj as IAddressOutExpression;
            if (expression == null)
                return false;

            return Expression.Equals(expression.Expression);
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
            return null;
        }
    }
}