// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// A Literal expression
    /// </summary>
    internal class XLiteralExpression : XExpression, ILiteralExpression
    {
        #region Fields

        private object val;

        #endregion

        #region ILiteralExpression Members

        /// <summary>
        /// The literal value
        /// </summary>
        public object Value
        {
            get { return this.val; }
            set { this.val = value; }
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

            ILiteralExpression expression = obj as ILiteralExpression;
            if (expression == null)
                return false;

            return this.Value.Equals(expression.Value);
        }

        public override int GetHashCode()
        {
            return this.Value.GetHashCode();
        }

        #endregion Object Overrides

        /// <summary>
        /// Get expression type
        /// </summary>
        public override Type GetExpressionType()
        {
            if (this.Value == null) return typeof (object);
            else return this.Value.GetType();
        }
    }
}