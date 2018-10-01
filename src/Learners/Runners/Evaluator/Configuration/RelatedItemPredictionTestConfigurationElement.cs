// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Xml.Serialization;

    /// <summary>
    /// The configuration element that describes the test for related item prediction.
    /// </summary>
    [Serializable]
    [XmlType(TypeName = "PredictRelatedItems")]
    public class RelatedItemPredictionTestConfigurationElement : RecommenderTestConfigurationElement
    {
        /// <summary>
        /// Gets or sets the maximum number of items to relate to an item during the test.
        /// </summary>
        public int? MaxRelatedItemCount { get; set; }

        /// <summary>
        /// Gets or sets the minimum number of common ratings between two items.
        /// </summary>
        public int? MinCommonRatingCount { get; set; }

        /// <summary>
        /// Gets or sets the minimum size of the related item pool for a single item.
        /// </summary>
        public int? MinRelatedItemPoolSize { get; set; }
        
        /// <summary>
        /// Creates an instance of <see cref="RelatedItemPredictionTestConfigurationElement"/> using the settings from this configuration element.
        /// </summary>
        /// <returns>An instance of <see cref="RelatedItemPredictionTestConfigurationElement"/>.</returns>
        public override RecommenderTest Create()
        {
            return new RelatedItemPredictionTest(this.MaxRelatedItemCount.Value, this.MinCommonRatingCount.Value, this.MinRelatedItemPoolSize.Value);
        }
    }
}