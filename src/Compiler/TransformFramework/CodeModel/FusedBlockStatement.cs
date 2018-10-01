// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// A block of IForStatements that will be merged.
    /// </summary>
    internal class FusedBlockStatement : XWhileStatement, IFusedBlockStatement
    {
        /// <summary>
        /// Create a fused block statement.
        /// </summary>
        /// <param name="condition"></param>
        internal FusedBlockStatement(IExpression condition)
        {
            this.Condition = condition;
        }
    }
}

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// A block of IForStatements that will be merged.
    /// </summary>
    internal interface IFusedBlockStatement : IWhileStatement
    {
    }
}