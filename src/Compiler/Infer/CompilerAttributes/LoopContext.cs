// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.Attributes
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Represents a loop context i.e. the set of loops that an expression occurs in.
    /// </summary>
    internal class LoopContext : ICompilerAttribute
    {
        /// <summary>
        /// Helps recognize code patterns
        /// </summary>
        private static CodeRecognizer Recognizer = CodeRecognizer.Instance;

        /// <summary>
        /// Loops that contain the expression, outermost first.
        /// </summary>
        internal List<IForStatement> loops; // = new List<IForStatement>();

        /// <summary>
        /// The loop variables for all contained loops
        /// </summary>
        internal List<IVariableDeclaration> loopVariables = new List<IVariableDeclaration>();

        /// <summary>
        /// Creates a loop context, given the current transform context.
        /// </summary>
        internal LoopContext(BasicTransformContext context) : this(context.FindAncestors<IForStatement>())
        {
        }

        /// <summary>
        /// Creates a loop context, given the current transform context.
        /// </summary>
        internal LoopContext(List<IForStatement> loops)
        {
            this.loops = loops;
            foreach (IForStatement loop in loops)
            {
                loopVariables.Add(Recognizer.LoopVariable(loop));
            }
        }

        /// <summary>
        /// Gets the reference loop context for a reference to a local variable.  A reference loop context
        /// is the set of loops that a variable reference occurs in, less any loops that the variable declaration
        /// occurred in.
        /// </summary>
        /// <returns></returns>
        internal RefLoopContext GetReferenceLoopContext(BasicTransformContext context)
        {
            List<IForStatement> loops = context.FindAncestors<IForStatement>();
            RefLoopContext rlc = new RefLoopContext();
            try
            {
                foreach (IForStatement loop in loops)
                {
                    IVariableDeclaration loopVar = Recognizer.LoopVariable(loop);
                    if (loopVariables.Contains(loopVar)) continue;
                    rlc.loopVariables.Add(loopVar);
                    rlc.loops.Add(loop);
                }
            }
            catch (Exception ex)
            {
                context.Error("Could not get loop index variables", ex);
            }
            return rlc;
        }

        public override string ToString()
        {
            return "LoopContext" + Util.CollectionToString(loopVariables);
        }
    }

    /// <summary>
    /// Represents a reference loop context i.e. the set of loops that a variable reference 
    /// occurs in, less any loops that the variable declaration occurred in.
    /// </summary>
    internal class RefLoopContext
    {
        // Loops that the reference is in that the declaration isn't
        internal List<IForStatement> loops = new List<IForStatement>();
        // Indices for the above loops
        internal List<IVariableDeclaration> loopVariables = new List<IVariableDeclaration>();

        public override string ToString()
        {
            return "RefLoopContext" + Util.CollectionToString(loopVariables);
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}