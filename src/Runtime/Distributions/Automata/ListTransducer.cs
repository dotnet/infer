// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Math;
using System.Collections.Generic;
namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    /// <summary>
    /// Represents a transducer defined on pairs of lists.
    /// </summary>
    public class ListTransducer<TList, TElement,TElementDist> :
        Transducer<TList, TElement, TElementDist, ListManipulator<TList,TElement>, 
        ListAutomaton<TList, TElement, TElementDist>, ListTransducer<TList, TElement, TElementDist>>
               where TElementDist : class, IImmutableDistribution<TElement, TElementDist>, CanGetLogAverageOf<TElementDist>, CanComputeProduct<TElementDist>, CanCreatePartialUniform<TElementDist>, SummableExactly<TElementDist>, Sampleable<TElement>, new()
        where TList: class, IList<TElement>, new()


    {
    }

    /// <summary>
    /// Represents a transducer defined on pairs of lists.
    /// </summary>
    public class ListTransducer<TElement, TElementDist> :
        Transducer<List<TElement>, TElement, TElementDist, ListManipulator<List<TElement>, TElement>,
        ListAutomaton<TElement, TElementDist>, ListTransducer<TElement, TElementDist>>
               where TElementDist : class, IImmutableDistribution<TElement, TElementDist>, CanGetLogAverageOf<TElementDist>, CanComputeProduct<TElementDist>, CanCreatePartialUniform<TElementDist>, SummableExactly<TElementDist>, Sampleable<TElement>, new()
    {
    }
}