// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// A 'this' reference expression
    /// </summary>
    internal class XThisReferenceExpression : XExpression, IThisReferenceExpression
    {
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

            IThisReferenceExpression expression = obj as IThisReferenceExpression;
            if (expression == null)
                return false;

            return true;
        }

        public override int GetHashCode()
        {
            return 4; // all ThisReferenceExpressions have the same hash code
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