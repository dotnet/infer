// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

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
    /// Represents a repeat context i.e. the set of repeat blocks that an expression occurs in.
    /// </summary>
    internal class RepeatContext : ICompilerAttribute
    {
        /// <summary>
        /// Helps recognize code patterns
        /// </summary>
        private static CodeRecognizer Recognizer = CodeRecognizer.Instance;

        /// <summary>
        /// Repeat blocks that contain the expression, outermost first.
        /// </summary>
        internal List<IRepeatStatement> repeats; // = new List<IForStatement>();

        /// <summary>
        /// The repeat counts for all contained repeat blocks
        /// </summary>
        internal List<IExpression> repeatCounts = new List<IExpression>();


        /// <summary>
        /// Creates a repeat context, given the current transform context.
        /// </summary>
        internal RepeatContext(BasicTransformContext context) : this(context.FindAncestors<IRepeatStatement>())
        {
        }

        /// <summary>
        /// Creates a repeat context, given the current transform context.
        /// </summary>
        internal RepeatContext(List<IRepeatStatement> repeats)
        {
            this.repeats = repeats;
            foreach (IRepeatStatement rep in repeats)
            {
                repeatCounts.Add(rep.Count);
            }
        }

        /// <summary>
        /// Gets the reference loop context for a reference to a local variable.  A reference loop context
        /// is the set of loops that a variable reference occurs in, less any loops that the variable declaration
        /// occurred in.
        /// </summary>
        /// <returns></returns>
        internal RefRepeatContext GetReferenceRepeatContext(BasicTransformContext context)
        {
            List<IRepeatStatement> reps = context.FindAncestors<IRepeatStatement>();
            // Make a cloned list of repeat counts and remove when found 
            var rcs = new List<IExpression>(repeatCounts);
            RefRepeatContext rlc = new RefRepeatContext();
            foreach (IRepeatStatement rep in reps)
            {
                IExpression repCount = rep.Count;
                int k = rcs.IndexOf(repCount);
                if (k != -1)
                {
                    rcs.RemoveAt(k); // remove this count so we don't use it again.
                    continue;
                }
                rlc.repeatCounts.Add(repCount);
                rlc.repeats.Add(rep);
            }
            return rlc;
        }

        public override string ToString()
        {
            return "RepeatContext" + Util.CollectionToString(repeatCounts);
        }
    }

    /// <summary>
    /// Represents a reference loop context i.e. the set of loops that a variable reference 
    /// occurs in, less any loops that the variable declaration occurred in.
    /// </summary>
    internal class RefRepeatContext
    {
        // Repeat blocks that the reference is in that the declaration isn't
        internal List<IRepeatStatement> repeats = new List<IRepeatStatement>();
        // Counts for the above repeat blocks
        internal List<IExpression> repeatCounts = new List<IExpression>();

        public override string ToString()
        {
            return "RefLoopContext" + Util.CollectionToString(repeatCounts);
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}