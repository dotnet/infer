// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Represent an array of index expressions applied to an array expression
    /// </summary>
    internal class XArrayIndexerExpression : XExpression, IArrayIndexerExpression
    {
        #region Fields

        private IList<IExpression> indices;
        private IExpression target;

        #endregion

        #region IArrayIndexerExpression Members

        public IList<IExpression> Indices
        {
            get
            {
                if (indices == null)
                    indices = new List<IExpression>();
                return indices;
            }
        }

        public IExpression Target
        {
            get { return this.target; }
            set { this.target = value; }
        }

        #endregion

        #region Object Overrides

        public override string ToString()
        {
            // RTODO
            return base.ToString();
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            IArrayIndexerExpression expression = obj as IArrayIndexerExpression;
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
            if (this.Target == null) return null;
            Type arrayType = this.Target.GetExpressionType();
            if (arrayType == null)
                return null;
            if (arrayType.IsArray)
                return arrayType.GetElementType();
            else
                return Utilities.Util.GetElementType(arrayType);
        }
    }
}