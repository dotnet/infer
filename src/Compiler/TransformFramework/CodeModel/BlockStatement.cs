// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// A block of statements
    /// </summary>
    internal class XBlockStatement : XStatement, IBlockStatement
    {
        #region Fields

        private IList<IStatement> statements;

        #endregion

        #region IBlockStatement Members

        /// <summary>
        /// The statements in the block
        /// </summary>
        public IList<IStatement> Statements
        {
            get
            {
                if (this.statements == null)
                    this.statements = new List<IStatement>();
                return this.statements;
            }
        }

        #endregion

        #region Object Overrides

        public override string ToString()
        {
            // Base statement will deal with this correctly
            return base.ToString();
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            IBlockStatement statement = obj as IBlockStatement;
            if (statement == null ||
                Statements.Count != statement.Statements.Count)
                return false;

            for (int i = 0; i < Statements.Count; i++)
                if (!(Statements[i].Equals(statement.Statements[i])))
                    return false;

            return true;
        }

        public override int GetHashCode()
        {
            int hash = Hash.Start;
            foreach (IStatement st in this.Statements)
            {
                hash = Hash.Combine(hash, st.GetHashCode());
            }
            return hash;
        }

        #endregion Object Overrides
    }
}