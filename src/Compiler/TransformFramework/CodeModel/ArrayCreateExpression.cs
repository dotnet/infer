// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Represents an array creation expression
    /// </summary>
    internal class XArrayCreateExpression : XExpression, IArrayCreateExpression
    {
        #region Fields

        private IList<IExpression> dimensions;
        private IBlockExpression initializer;
        private IType type;

        #endregion

        #region IArrayCreateExpression Members

        /// <summary>
        /// Dimension array
        /// </summary>
        public IList<IExpression> Dimensions
        {
            get
            {
                if (this.dimensions == null)
                    this.dimensions = new List<IExpression>();
                return this.dimensions;
            }
        }

        /// <summary>
        /// Array initializer
        /// </summary>
        public IBlockExpression Initializer
        {
            get { return this.initializer; }
            set { this.initializer = value; }
        }

        /// <summary>
        /// Array element type
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
            // RTODO
            return base.ToString();
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            IArrayCreateExpression expression = obj as IArrayCreateExpression;
            if (expression == null ||
                (!Type.Equals(expression.Type)) ||
                (Initializer != null && !Initializer.Equals(expression.Initializer)) ||
                Dimensions.Count != expression.Dimensions.Count)
                return false;

            for (int i = 0; i < Dimensions.Count; i++)
                if (!(Dimensions[i].Equals(expression.Dimensions[i])))
                    return false;

            return true;
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
            int rank = this.Dimensions.Count;
            Type tp = this.Type.DotNetType;
            if (rank == 0)
                return tp;
            else if (rank == 1)
                return tp.MakeArrayType(); // bizarrely, gives different result to MakeArrayType(1);
            else
                return tp.MakeArrayType(rank);
        }
    }
}