// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Anonymous method expression
    /// </summary>
    internal class XAnonymousMethodExpression : XExpression, IAnonymousMethodExpression
    {
        #region Fields

        private IBlockStatement body;
        private IType delegateType;
        private IList<IParameterDeclaration> parameters;
        private IMethodReturnType returnType;

        #endregion Fields

        #region IAnonymousMethodExpression Members

        /// <summary>
        /// Body of the anonymous expression
        /// </summary>
        public IBlockStatement Body
        {
            get { return this.body; }
            set { this.body = value; }
        }

        /// <summary>
        /// Anonymous expression delegate type
        /// </summary>
        public IType DelegateType
        {
            get { return this.delegateType; }
            set { this.delegateType = value; }
        }

        /// <summary>
        /// Method parameters
        /// </summary>
        public IList<IParameterDeclaration> Parameters
        {
            get
            {
                if (this.parameters == null)
                {
                    this.parameters = new List<IParameterDeclaration>();
                }
                return this.parameters;
            }
        }

        /// <summary>
        /// Anonymous method return type
        /// </summary>
        public IMethodReturnType ReturnType
        {
            get { return this.returnType; }
            set { this.returnType = value; }
        }

        #endregion IAnonymousMethodExpression

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

            IAnonymousMethodExpression expression = obj as IAnonymousMethodExpression;
            if (expression == null)
                return false;

            // Check that the parameter count is the same
            if (expression.Parameters.Count != this.Parameters.Count)
                return false;

            // Check that the parameter types are the same
            for (int i = 0; i < this.Parameters.Count; i++)
            {
                if (!this.Parameters[i].ParameterType.Equals(expression.Parameters[i].ParameterType))
                    return false;
            }

            // Check that the body is the same
            if (!this.Body.Equals(expression.Body))
                return false;

            return true;
        }

        public override int GetHashCode()
        {
            return this.DelegateType.GetHashCode();
        }

        #endregion Object Overrides

        /// <summary>
        /// Get expression type
        /// </summary>
        public override Type GetExpressionType()
        {
            return delegateType.DotNetType;
        }
    }
}