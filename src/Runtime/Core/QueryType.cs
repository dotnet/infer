// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic
{
    /// <summary>
    /// Type of inference query
    /// </summary>
    public class QueryType
    {
        /// <summary>
        /// Name of the query type
        /// </summary>
        public string Name;

        public override string ToString()
        {
            return Name;
        }
    }
}