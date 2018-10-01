// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic
{
    /// <summary>
    /// Static query types
    /// </summary>
    public static class QueryTypes
    {
        /// <summary>
        /// The default query type - returns the marginal.
        /// </summary>
        public static QueryType Marginal = new QueryType {Name = "Marginal"};

        /// <summary>
        /// This returns the samples for sampling algorithms
        /// </summary>
        public static QueryType Samples = new QueryType {Name = "Samples"};

        /// <summary>
        /// For sampling algorithms, this returns the list of distributions from which the samples were drawn.
        /// </summary>
        public static QueryType Conditionals = new QueryType {Name = "Conditionals"};

        /// <summary>
        /// Returns the marginal distribution divided by the prior distribution, leaving only the data likelihood
        /// </summary>
        public static QueryType MarginalDividedByPrior = new QueryType {Name = "MarginalDividedByPrior"};

        // when adding new QueryTypes, make sure that no QueryType is a suffix of another QueryType, otherwise you may get name clashes in generated code
    }
}