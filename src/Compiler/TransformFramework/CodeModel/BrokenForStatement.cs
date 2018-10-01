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
    /// An IForStatement whose body mutates the loop counter, or contains a "break" or "continue" statement.
    /// </summary>
    internal class BrokenForStatement : XForStatement, IBrokenForStatement
    {
        /// <summary>
        /// Create a broken copy of a "for" statement.
        /// </summary>
        /// <param name="ifs"></param>
        internal BrokenForStatement(IForStatement ifs)
        {
            this.Initializer = ifs.Initializer;
            this.Condition = ifs.Condition;
            this.Increment = ifs.Increment;
            this.Body = ifs.Body;
        }
    }
}

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// An IForStatement whose body mutates the loop counter, or contains a "break" or "continue" statement.
    /// </summary>
    internal interface IBrokenForStatement : IForStatement
    {
    }
}