// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Base reference expression
    /// </summary>
    internal class XBaseReferenceExpression : XExpression, IBaseReferenceExpression
    {
        #region Object Overrides

        public override string ToString()
        {
            return base.ToString();
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            IBaseReferenceExpression expression = obj as IBaseReferenceExpression;
            if (expression == null)
                return false;
            else
                return true;
        }

        public override int GetHashCode()
        {
            // hash code must be a constant due to behaviour of Equals
            return 131071;
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