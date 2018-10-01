// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    internal class XTryCatchFinallyStatement : ITryCatchFinallyStatement
    {
        #region Fields

        private IList<ICatchClause> catchClauses;
        private IBlockStatement faultStatements;
        private IBlockStatement finalStatements;
        private IBlockStatement tryStatements;

        #endregion

        #region ITryCatchFinallyStatement Members

        /// <summary>
        /// The catch clauses
        /// </summary>
        public IList<ICatchClause> CatchClauses
        {
            get
            {
                if (this.catchClauses == null)
                {
                    this.catchClauses = new List<ICatchClause>();
                }
                return this.catchClauses;
            }
        }

        /// <summary>
        /// Default catch statement
        /// </summary>
        public IBlockStatement Fault
        {
            get { return this.faultStatements; }
            set { this.faultStatements = value; }
        }

        /// <summary>
        /// Finally statement
        /// </summary>
        public IBlockStatement Finally
        {
            get { return this.finalStatements; }
            set { this.finalStatements = value; }
        }

        public IBlockStatement Try
        {
            get { return this.tryStatements; }
            set { this.tryStatements = value; }
        }

        #endregion

        #region Object Overrides

        public override string ToString()
        {
            // Base statement deals with this
            return base.ToString();
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            ITryCatchFinallyStatement statement = obj as ITryCatchFinallyStatement;
            if (statement == null)
                return false;

            if (this.CatchClauses.Count != statement.CatchClauses.Count)
                return false;

            for (int i = 0; i < CatchClauses.Count; i++)
                if (!(CatchClauses[i].Equals(statement.CatchClauses[i])))
                    return false;

            if (!this.Try.Equals(statement.Try))
                return false;

            if (!this.Fault.Equals(statement.Fault))
                return false;

            if (!this.Finally.Equals(statement.Finally))
                return false;

            return true;
        }

        public override int GetHashCode()
        {
            return this.Try.GetHashCode();
        }

        #endregion Object Overrides
    }
}