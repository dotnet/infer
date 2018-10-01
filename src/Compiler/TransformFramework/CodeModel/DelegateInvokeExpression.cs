// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Method invoke expression
    /// </summary>
    internal class XDelegateInvokeExpression : XExpression, IDelegateInvokeExpression
    {
        #region Fields

        private IList<IExpression> arguments;
        private IExpression target;

        #endregion

        #region IDelegateInvokeExpression Members

        public IExpression Target
        {
            get { return this.target; }
            set { this.target = value; }
        }

        /// <summary>
        /// Arguments of the method
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

            IDelegateInvokeExpression expression = obj as IDelegateInvokeExpression;
            if (expression == null ||
                !Target.Equals(expression.Target) ||
                Arguments.Count != expression.Arguments.Count)
                return false;

            for (int i = 0; i < Arguments.Count; i++)
                if (!(Arguments[i].Equals(expression.Arguments[i])))
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
            IEventReferenceExpression iere = this.Target as IEventReferenceExpression;
            return iere.Event.EventType.DotNetType;
        }
    }
}