// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.IO;

namespace Microsoft.ML.Probabilistic.Compiler
{
    /// <summary>
    /// This class is used by a language writer to build a tree of source fragments
    /// </summary>
    internal class SourceNode
    {
        /// <summary>
        /// The start string for the code element
        /// </summary>
        public string StartString;

        /// <summary>
        /// The end string for the code element
        /// </summary>
        public string EndString;

        /// <summary>
        /// The element in the abstract syntax tree corresponding to this source fragment
        /// </summary>
        public object ASTElement;

        /// <summary>
        /// Children nodes
        /// </summary>
        public List<SourceNode> Children;

        /// <summary>
        /// Clear the node
        /// </summary>
        public void Clear()
        {
            StartString = "";
            EndString = "";
            ASTElement = null;
            if (Children != null)
                Children.Clear();
            Children = null;
        }

        /// <summary>
        /// Default constructor
        /// </summary>
        public SourceNode()
        {
            Clear();
        }

        /// <summary>
        /// Construct from start string
        /// </summary>
        /// <param name="start"></param>
        public SourceNode(string start)
            : this()
        {
            StartString = start;
        }

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="start">start string</param>
        /// <param name="end">end string</param>
        /// <param name="element">element</param>
        public SourceNode(string start, string end, object element)
            : this(start)
        {
            EndString = end;
            ASTElement = element;
        }

        /// <summary>
        /// Display the source node as a string in a given language
        /// </summary>
        /// <returns></returns>
        public string ToString(ILanguageWriter writer)
        {
            StringWriter sw = new StringWriter();
            sw.Write(this.StartString);
            if (this.Children != null)
            {
                foreach (SourceNode child in this.Children)
                    LanguageWriter.WriteSourceNode(sw, child);
            }
            sw.Write(this.EndString);
            return sw.ToString();
        }
    }
}