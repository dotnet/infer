// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.IO;
    using System.Xml.Serialization;
    using Microsoft.ML.Probabilistic.Learners.Runners.DatasetGenerators;

    /// <summary>
    /// The configuration element that describes a recommendation dataset that has to be generated.
    /// </summary>
    [Serializable]
    [XmlType(TypeName = "RecommenderGeneratedDataset")]
    public class RecommenderGeneratedDatasetConfigurationElement : RecommenderDatasetConfigurationElement
    {
        /// <summary>
        /// Gets or sets the generator of dataset.
        /// </summary>
        public string Generator { get; set; }

        /// <summary>
        /// Generate dataset if it necessary and loads the dataset using the settings from this configuration element.
        /// </summary>
        /// <returns>The loaded dataset.</returns>
        public override RecommenderDataset Load()
        {
            if (!File.Exists(FileName))
            {
                Type t = Type.GetType(Generator);
                if(t == null)
                {
                    throw new InvalidOperationException($"{Generator} type is undefined");
                }
                IDatasetGenerator generator = (IDatasetGenerator)Activator.CreateInstance(t);
                generator.Generate(FileName);
            }

            return RecommenderDataset.Load(this.FileName);
        }
    }
}
