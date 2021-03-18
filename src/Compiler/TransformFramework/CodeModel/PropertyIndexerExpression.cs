// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Property index expression
    /// </summary>
    internal class XPropertyIndexerExpression : XExpression, IPropertyIndexerExpression
    {
        #region Fields

        private IPropertyReferenceExpression target;
        private IList<IExpression> indices;

        #endregion

        #region IPropertyIndexerExpression Members

        /// <summary>
        /// Expressions for the indices
        /// </summary>
        public IList<IExpression> Indices
        {
            get
            {
                if (indices == null)
                    indices = new List<IExpression>();
                return indices;
            }
        }

        /// <summary>
        /// Property reference expression
        /// </summary>
        public IPropertyReferenceExpression Target
        {
            get { return this.target; }
            set { this.target = value; }
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

            IPropertyIndexerExpression expression = obj as IPropertyIndexerExpression;
            if (expression == null ||
                (!Target.Equals(expression.Target)) ||
                Indices.Count != expression.Indices.Count)
                return false;

            for (int i = 0; i < Indices.Count; i++)
                if (!(Indices[i].Equals(expression.Indices[i])))
                    return false;

            return true;
        }

        public override int GetHashCode()
        {
            return this.Target.GetHashCode();
        }

        #endregion Object Overrides

        /// <summary>
        /// Get expression type
        /// </summary>
        public override Type GetExpressionType()
        {
            return Target.Property.PropertyType.DotNetType;
        }
    }
}