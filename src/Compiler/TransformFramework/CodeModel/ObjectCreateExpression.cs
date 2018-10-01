// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Object create expression
    /// </summary>
    internal class XObjectCreateExpression : XExpression, IObjectCreateExpression
    {
        #region Fields

        private IList<IExpression> arguments;
        private IMethodReference constructor;
        private IType type;
        private IBlockExpression initializer;

        #endregion

        #region IObjectCreateExpression Members

        /// <summary>
        /// Expressions for the constructor arguments
        /// </summary>
        public IList<IExpression> Arguments
        {
            get
            {
                if (this.arguments == null)
                    this.arguments = new List<IExpression>();
                return this.arguments;
            }
        }

        /// <summary>
        /// The constructor method reference
        /// </summary>
        public IMethodReference Constructor
        {
            get { return this.constructor; }
            set { this.constructor = value; }
        }

        /// <summary>
        /// Initializer
        /// </summary>
        public IBlockExpression Initializer
        {
            get { return this.initializer; }
            set { this.initializer = value; }
        }

        /// <summary>
        /// Type of object being constructed
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

            IObjectCreateExpression expression = obj as IObjectCreateExpression;
            if (expression == null)
                return false;

            if (!Constructor.Equals(expression.Constructor))
                return false;

            if (Initializer == null)
            {
                if (expression.Initializer != null)
                    return false;
            }
            else if (!Initializer.Equals(expression.Initializer))
                return false;

            if (Arguments.Count != expression.Arguments.Count)
                return false;

            for (int i = 0; i < Arguments.Count; i++)
                if (!(Arguments[i].Equals(expression.Arguments[i])))
                    return false;

            return true;
        }

        public override int GetHashCode()
        {
            int hash = Hash.Start;
            hash = Hash.Combine(hash, (Constructor == null) ? 0 : Constructor.GetHashCode());
            hash = Hash.Combine(hash, (Initializer == null) ? 0 : Initializer.GetHashCode());
            foreach (IExpression arg in Arguments)
            {
                hash = Hash.Combine(hash, arg.GetHashCode());
            }
            return hash;
        }

        #endregion Object Overrides

        /// <summary>
        /// Get expression type
        /// </summary>
        public override Type GetExpressionType()
        {
            return this.Type.DotNetType;
        }
    }
}