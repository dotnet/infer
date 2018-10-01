// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// A switch statement
    /// </summary>
    internal class XSwitchStatement : XStatement, ISwitchStatement
    {
        #region Fields

        private IList<ISwitchCase> cases;
        private IExpression expression;

        #endregion

        #region ISwitchStatement Members

        /// <summary>
        /// The collection of cases
        /// </summary>
        public IList<ISwitchCase> Cases
        {
            get
            {
                if (this.cases == null)
                    this.cases = new List<ISwitchCase>();
                return this.cases;
            }
        }

        /// <summary>
        /// The expression
        /// </summary>
        public IExpression Expression
        {
            get { return this.expression; }
            set { this.expression = value; }
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

            ISwitchStatement statement = obj as ISwitchStatement;
            if (statement == null ||
                (!Expression.Equals(statement.Expression)) ||
                Cases.Count != statement.Cases.Count)
                return false;

            for (int i = 0; i < Cases.Count; i++)
                if (!(Cases[i].Equals(statement.Cases[i])))
                    return false;

            return true;
        }

        public override int GetHashCode()
        {
            return this.Expression.GetHashCode();
        }

        #endregion Object Overrides
    }
}