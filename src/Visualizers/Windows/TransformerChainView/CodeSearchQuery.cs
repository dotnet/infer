// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Text.RegularExpressions;

namespace Microsoft.ML.Probabilistic.Compiler.Visualizers
{
    /// <summary>
    /// A search query on lines of code.
    /// </summary>
    internal class CodeSearchQuery
    {
        internal List<string> Terms { get; set; }
        private Regex regex;

        private CodeSearchQuery()
        {
            Terms = new List<string>();
        }

        /// <summary>
        /// Creates a code search query from a query string.
        /// </summary>
        /// <param name="queryString"></param>
        /// <returns></returns>
        internal static CodeSearchQuery FromQueryString(string queryString)
        {
            bool inQuotes = false;
            var sb = new StringBuilder();
            var query = new CodeSearchQuery();
            foreach (var ch in queryString)
            {
                if (ch == '"')
                {
                    inQuotes = !inQuotes;
                }
                else if ((ch == ' ') && !inQuotes)
                {
                    query.AddTerm(sb.ToString());
                    sb.Clear();
                    continue;
                }
                sb.Append(ch);                
            }
            if (sb.Length > 0)
            {
                query.AddTerm(sb.ToString());
            }
            return query;
        }

        internal void AddTerm(string term)
        {
            if (string.IsNullOrWhiteSpace(term)) return;
            var qt = term.Trim();
            Terms.Add(qt);
            regex = null;
        }

        private void SetRegex()
        {
            StringBuilder sb = new StringBuilder();
            foreach (var term in Terms)
            {
                string s = Regex.Escape(term);
                // quotes force matching at a word boundary
                s = s.Replace("\"", @"\b");
                // * matches any string
                s = s.Replace("\\*", @".*");
                if (sb.Length > 0)
                    sb.Append('|');
                sb.Append(s);
            }
            regex = new Regex(sb.ToString(), RegexOptions.CultureInvariant | RegexOptions.IgnoreCase);
        }

        internal MatchCollection Matches(LineOfCode line)
        {
            if (regex == null)
                SetRegex();
            return regex.Matches(line.Text);
        }

        internal string ToQueryString()
        {
            var sb = new StringBuilder();
            foreach (var term in Terms)
            {
                if (sb.Length > 0) sb.Append(" ");
                sb.Append(term);
            }
            return sb.ToString();
        }
    }
}
