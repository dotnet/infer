// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Type reference expression
    /// </summary>
    internal class XTypeReferenceExpression : XExpression, ITypeReferenceExpression
    {
        #region Fields

        private ITypeReference type;

        #endregion

        #region ITypeReferenceExpression Members

        /// <summary>
        /// The type referecne
        /// </summary>
        public ITypeReference Type
        {
            get { return this.type; }
            set
            {
                if (this.type != value)
                    this.type = value;
            }
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

            ITypeReferenceExpression expression = obj as ITypeReferenceExpression;
            if (expression == null)
                return false;

            return this.Type.Equals(expression.Type);
        }

        public override int GetHashCode()
        {
            if (this.Type == null)
                return 0;
            else
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