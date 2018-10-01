// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Xml.Serialization;

    /// <summary>
    /// The configuration element that describes a recommendation dataset.
    /// </summary>
    [Serializable]
    [XmlType(TypeName = "RecommenderDataset")]
    [XmlInclude(typeof(RecommenderGeneratedDatasetConfigurationElement))]
    public class RecommenderDatasetConfigurationElement : ConfigurationElement
    {
        /// <summary>
        /// Gets or sets the name of the dataset file.
        /// </summary>
        public string FileName { get; set; }

        /// <summary>
        /// Loads the dataset using the settings from this configuration element.
        /// </summary>
        /// <returns>The loaded dataset.</returns>
        public virtual RecommenderDataset Load()
        {
            return RecommenderDataset.Load(this.FileName);
        }
    }
}
