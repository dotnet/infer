// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Collections.Generic;

    using Math;
    using Microsoft.ML.Probabilistic.Collections;

    /// <summary>
    /// Interface to allow untyped access to collection distribution
    /// </summary>
    public interface ICollectionDistribution : IDistribution
    {
        /// <summary>
        /// Returns the count of known elements in collection distribution.
        /// </summary>
        int GetElementsCount();

        /// <summary>
        /// Returns the list of elements' distributions.
        /// </summary>
        List<IDistribution> GetElementsUntyped();

        /// <summary>
        /// Product of two collection distributions which also return element mapping information.
        /// </summary>
        /// <param name="a">Collection distribution</param>
        /// <param name="b">Collection distribution</param>
        /// <param name="elementMapping">Element mapping information</param>
        void SetToProduct(ICollectionDistribution a, ICollectionDistribution b, out CollectionElementMappingInfo elementMapping);
    }

    /// <summary>
    /// Element mapping information for the product of collection distributions
    /// </summary>
    public sealed class CollectionElementMappingInfo
    {
        public CollectionElementMappingInfo(List<(ReadOnlyArray<int>, ReadOnlyArray<int>)> elementMapping)
        {
            this.ElementMapping = elementMapping;
        }

        /// <summary>
        /// The evidence associated with the elements of collection distribution
        /// </summary>
        public List<(ReadOnlyArray<int>, ReadOnlyArray<int>)> ElementMapping { get; }
    }

    /// <summary>
    /// Collection distribution interface
    /// </summary>
    /// <typeparam name="TElement">Element domain type</typeparam>
    /// <typeparam name="TElementDist">Element distribution type</typeparam>
    public interface ICollectionDistribution<TElement, TElementDist> :
        ICollectionDistribution
        where TElementDist : IDistribution<TElement>
    {
        /// <summary>
        /// Returns the list of elements' distributions.
        /// </summary>
        List<TElementDist> GetElements();

        /// <summary>
        /// Transforms elements of collection by changing their distribution type.
        /// The element domain type remains unchanged.
        /// </summary>
        /// <typeparam name="TNewElementDist">New element distribution type</typeparam>
        ICollectionDistribution<TElement, TNewElementDist> TransformElements<TNewElementDist>(Func<TElementDist, TNewElementDist> transformFunc)
            where TNewElementDist :
                IDistribution<TElement>,
                CanGetLogAverageOf<TNewElementDist>,
                Sampleable<TElement>,
                SettableToProduct<TNewElementDist>;
    }
}