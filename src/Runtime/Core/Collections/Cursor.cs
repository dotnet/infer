// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Probabilistic.Collections
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    public interface ICursor : ICloneable
    {
        /// <summary>
        /// The position of the cursor in its source array.
        /// </summary>
        /// <remarks>
        /// Position is measured in the same units as Count.
        /// The instance data spans locations <c>Start, ..., Start+Count-1</c> in 
        /// the source array.
        /// </remarks>
        int Start { get; set; }

        /// <summary>
        /// The number of positions in the source array that one instance consumes.
        /// </summary>
        /// <remarks>
        /// The cursor can be advanced to the next instance via
        /// <c>Start = Start + Count</c>.
        /// </remarks>
        int Count { get; }

        /// <summary>
        /// Point the cursor at a new source array.
        /// </summary>
        /// <remarks>
        /// The source array is allocated to have nRecords * Count positions.
        /// </remarks>
        void CreateSourceArray(int nRecords);

        /// <summary>
        /// Make a new cursor object having the same source array, at the same position.
        /// </summary>
        ICursor ReferenceClone();
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}