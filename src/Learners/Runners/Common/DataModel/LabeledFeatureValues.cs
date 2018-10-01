// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Globalization;
    using System.Linq;
    using System.Text;

    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Math;

    /// <summary>
    /// Represents a label and its feature values.
    /// </summary>
    public class LabeledFeatureValues
    {
        #region Fields and constructors

        /// <summary>
        /// The tolerance for double comparison.
        /// </summary>
        private const double Tolerance = 1e-12;

        /// <summary>
        /// The default value of a feature whose name is specified, but whose value is not.
        /// </summary>
        private const double DefaultFeatureValue = 1.0;

        /// <summary>
        /// The character separating features in the specification of a feature vector.
        /// </summary>
        private const char FeatureSeparator = ' ';

        /// <summary>
        /// The character separating feature names from feature value.
        /// </summary>
        private const char NameValueSeparator = ':';

        /// <summary>
        /// Contains a zero-sparse representation of feature values.
        /// </summary>
        private readonly Dictionary<int, double> featureValues;

        /// <summary>
        /// Initializes a new instance of the <see cref="LabeledFeatureValues"/> class.
        /// </summary>
        public LabeledFeatureValues()
        {
            this.LabelDistribution = new LabelDistribution();
            this.FeatureSet = new IndexedSet<string>();
            this.featureValues = new Dictionary<int, double>();
        }

        #endregion

        #region Properties and methods

        /// <summary>
        /// Gets the label distribution associated with the feature values.
        /// </summary>
        public LabelDistribution LabelDistribution { get; private set; }

        /// <summary>
        /// Gets the bidirectional mapping from feature names to feature indexes.
        /// </summary>
        public IndexedSet<string> FeatureSet { get; private set; }

        /// <summary>
        /// Parses a specified string for a label and its feature values.
        /// </summary>
        /// <param name="specification">The string to be parsed for a label and its feature values.</param>
        /// <param name="labelSet">An optional bidirectional mapping from class labels to class indexes.</param>
        /// <param name="featureSet">An optional bidirectional mapping from feature names to feature indexes.</param>
        /// <returns>Labeled feature values equivalent to the given string specification.</returns>
        public static LabeledFeatureValues Parse(
            string specification,
            IndexedSet<string> labelSet = null,
            IndexedSet<string> featureSet = null)
        {
            var labeledFeatureValues = new LabeledFeatureValues();

            if (featureSet != null)
            {
                labeledFeatureValues.FeatureSet = featureSet;
            }

            // Parse label distribution
            labeledFeatureValues.LabelDistribution = LabelDistribution.Parse(specification, labelSet, out specification);

            // Parse feature values
            labeledFeatureValues.ParseFeatures(specification);

            return labeledFeatureValues;
        }

        /// <summary>
        /// Gets the feature values of the specified feature selection as a dense vector.
        /// </summary>
        /// <param name="featureSelection">An optional selection of features. Defaults to all features being selected.</param>
        /// <returns>The feature values as a dense vector.</returns>
        public Vector GetDenseFeatureVector(IndexedSet<string> featureSelection = null)
        {
            double[] denseValues;
            
            if (featureSelection == null)
            {
                // Return all feature values as a dense vector
                denseValues = new double[this.FeatureSet.Count];
                foreach (var feature in this.featureValues)
                {
                    denseValues[feature.Key] = feature.Value;
                }

                return Vector.FromArray(denseValues);
            }

            // Construct dense feature vector from selected indexes
            int featureCount = featureSelection.Count;
            denseValues = new double[featureCount];
            foreach (string featureName in featureSelection.Elements)
            {
                int featureIndex;
                if (this.FeatureSet.TryGetIndex(featureName, out featureIndex))
                {
                    if (this.featureValues.ContainsKey(featureIndex))
                    {
                        // Get original index
                        int selectedFeatureIndex;
                        if (!featureSelection.TryGetIndex(featureName, out selectedFeatureIndex))
                        {
                            throw new ArgumentException("Invalid feature selection.");
                        }

                        denseValues[selectedFeatureIndex] = this.featureValues[featureIndex];
                    }
                }
            }

            return Vector.FromArray(denseValues);
        }

        /// <summary>
        /// Gets the feature values of the specified feature selection as a sparse vector.
        /// </summary>
        /// <param name="featureSelection">An optional selection of features. Defaults to all features being selected.</param>
        /// <returns>The feature values as a sparse vector.</returns>
        public Vector GetSparseFeatureVector(IndexedSet<string> featureSelection = null)
        {
            List<ValueAtIndex<double>> sortedFeatures;

            if (featureSelection == null)
            {
                // Return all feature values as a dense vector
                sortedFeatures = this.featureValues.OrderBy(i => i.Key).Select(indexValuePair => new ValueAtIndex<double>(indexValuePair.Key, indexValuePair.Value)).ToList();
                return SparseVector.FromSparseValues(this.FeatureSet.Count, 0.0, sortedFeatures);
            }

            // Construct sparse feature vector from selected features
            int featureCount = featureSelection.Count;
            var selectedFeatures = new Dictionary<int, double>();
            foreach (string featureName in featureSelection.Elements)
            {
                int featureIndex;
                if (this.FeatureSet.TryGetIndex(featureName, out featureIndex))
                {
                    if (this.featureValues.ContainsKey(featureIndex))
                    {
                        // Get original index
                        int selectedFeatureIndex;
                        if (!featureSelection.TryGetIndex(featureName, out selectedFeatureIndex))
                        {
                            throw new ArgumentException("Invalid feature selection.");
                        }

                        selectedFeatures.Add(selectedFeatureIndex, this.featureValues[featureIndex]);
                    }
                }
            }

            sortedFeatures = 
                selectedFeatures.OrderBy(i => i.Key).Select(indexValuePair => new ValueAtIndex<double>(indexValuePair.Key, indexValuePair.Value)).ToList();
            return SparseVector.FromSparseValues(featureCount, 0.0, sortedFeatures);
        }

        /// <summary>
        /// Returns the labeled feature values as a string.
        /// </summary>
        /// <returns>The labeled feature values as a string.</returns>
        public override string ToString()
        {
            var builder = new StringBuilder();
            builder.Append(this.LabelDistribution);

            foreach (var featureValue in this.featureValues)
            {
                builder.AppendFormat(
                    "{0}{1}{2}{3}",
                    FeatureSeparator,
                    this.FeatureSet.GetElementByIndex(featureValue.Key),
                    NameValueSeparator,
                    featureValue.Value);
            }

            return builder.ToString();
        }

        #endregion

        #region Helper methods

        /// <summary>
        /// Parses a specified string for feature values.
        /// </summary>
        /// <param name="specification">The string to be parsed for feature values.</param>
        private void ParseFeatures(string specification)
        {
            if (string.IsNullOrEmpty(specification))
            {
                // No specification allowed
                return;
            }

            string trimmedSpecification = specification.Trim();
            if (this.IsDenseFeature(trimmedSpecification))
            {
                this.ParseDenseFeatures(trimmedSpecification);
            }
            else
            {
                this.ParseSparseFeatures(trimmedSpecification);
            }
        }

        /// <summary>
        /// Parses a specified string for a sparse representation of feature values.
        /// </summary>
        /// <param name="specification">The string to be parsed for a sparse representation of feature values.</param>
        private void ParseSparseFeatures(string specification)
        {
            if (string.IsNullOrEmpty(specification))
            {
                // No specification allowed
                return;
            }

            List<string> splits = specification.Split(FeatureSeparator).ToList();
            splits.RemoveAll(entry => entry == null || entry.Trim().Equals(string.Empty));
            if (splits.Count == 0)
            {
                // Nothing specified.
                return;
            }

            int i = 0;
            var sparseFeature = splits[i].Split(NameValueSeparator).ToList();
            sparseFeature.RemoveAll(entry => entry == null || entry.Trim().Equals(string.Empty));

            while (sparseFeature.Count == 1 || sparseFeature.Count == 2)
            {
                if (sparseFeature.Count == 1)
                {
                    // Only feature name is present, value defaults to 1
                    this.ParseFeature(sparseFeature[0]);
                }
                else
                {
                    // Both feature name and feature value are present
                    this.ParseFeature(sparseFeature[0], sparseFeature[1]);
                }

                if (++i >= splits.Count)
                {
                    break;
                }

                sparseFeature = splits[i].Split(NameValueSeparator).ToList();
                sparseFeature.RemoveAll(entry => entry == null || entry.Trim().Equals(string.Empty));
            }
        }

        /// <summary>
        /// Parses a specified string for a dense representation of feature values.
        /// </summary>
        /// <param name="specification">The string to be parsed for a dense representation of feature values.</param>
        private void ParseDenseFeatures(string specification)
        {
            var splits = specification.Split(FeatureSeparator).ToList();
            splits.RemoveAll(entry => entry == null || entry.Trim().Equals(string.Empty));

            if (string.IsNullOrEmpty(splits[0].Trim()))
            {
                // No specification allowed
                return;
            }

            for (int i = 0; i < splits.Count; i++)
            {
                this.ParseFeature((i + 1).ToString(CultureInfo.InvariantCulture), splits[i]);
            }
        }

        /// <summary>
        /// Parses the specified feature name and value.
        /// </summary>
        /// <param name="name">The feature name to parse.</param>
        /// <param name="value">The feature value to parse.</param>
        private void ParseFeature(string name, string value)
        {
            this.ParseFeature(name, this.ParseValue(value));
        }

        /// <summary>
        /// Parses a specified feature name given its value.
        /// </summary>
        /// <param name="name">The feature name to parse.</param>
        /// <param name="value">The feature value. Defaults to 1.</param>
        private void ParseFeature(string name, double value = DefaultFeatureValue)
        {
            name = this.ParseName(name);

            int index = this.FeatureSet.Add(name, false);

            if (this.featureValues.ContainsKey(index))
            {
                throw new InvalidFileFormatException("The feature name '" + name + "' must not be specified multiple times.");
            }

            if (Math.Abs(value) > Tolerance || double.IsNaN(value))
            {
                this.featureValues.Add(index, value);
            }
        }

        /// <summary>
        /// Parses a specified feature name.
        /// </summary>
        /// <param name="name">The feature name to parse.</param>
        /// <returns>The parsed feature name.</returns>
        private string ParseName(string name)
        {
            string featureName = name.Trim();
            if (string.IsNullOrEmpty(featureName))
            {
                throw new InvalidFileFormatException("The feature name must not be null, empty or whitespace.");
            }

            return featureName;
        }

        /// <summary>
        /// Parses a specified feature value.
        /// </summary>
        /// <param name="value">The feature value to parse.</param>
        /// <returns>The feature value.</returns>
        private double ParseValue(string value)
        {
            double featureValue;
            if (!double.TryParse(value.Trim(), out featureValue))
            {
                throw new InvalidFileFormatException("Unable to parse feature value '" + value.Trim() + "'.");
            }

            return featureValue;
        }

        /// <summary>
        /// Returns true if the specified feature string is a number and false otherwise.
        /// </summary>
        /// <param name="feature">The feature to test.</param>
        /// <returns>true if the specified feature string is a number and false otherwise.</returns>
        private bool IsDenseFeature(string feature)
        {
            Debug.Assert(feature != null, "The feature specification must not be null.");

            if (string.IsNullOrEmpty(feature))
            {
                return false;
            }

            if (feature.Contains(NameValueSeparator))
            {
                return false;
            }

            char first = feature.First();
            return char.IsDigit(first) || first.Equals('-') || first.Equals('+');
        }

        #endregion
    }
}
