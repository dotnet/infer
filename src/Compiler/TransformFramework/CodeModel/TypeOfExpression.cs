// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// 'typeof' expression
    /// </summary>
    internal class XTypeOfExpression : XExpression, ITypeOfExpression
    {
        #region Fields

        private IType type;

        #endregion

        #region ITypeOfExpression Members

        /// <summary>
        /// The type
        /// </summary>
        public IType Type
        {
            get { return this.type; }
            set { this.type = value; }
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

            ITypeOfExpression expression = obj as ITypeOfExpression;
            if (expression == null)
                return false;

            return this.Type.Equals(expression.Type);
        }

        public override int GetHashCode()
        {
            return this.Type.GetHashCode();
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