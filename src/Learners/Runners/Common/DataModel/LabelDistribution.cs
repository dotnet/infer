// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using System.Text;

    using Microsoft.ML.Probabilistic.Collections;

    /// <summary>
    /// A discrete distribution over a set of labels.
    /// </summary>
    public class LabelDistribution
    {
        #region Fields and constructors

        /// <summary>
        /// The character separating label-probability pairs in the specification of a distribution.
        /// </summary>
        public const char PointSeparator = ' ';

        /// <summary>
        /// The character separating a single label-probability pair in the specification of a distribution.
        /// </summary>
        public const char LabelProbabilitySeparator = '=';

        /// <summary>
        /// The tolerance for double comparison.
        /// </summary>
        private const double Tolerance = 1e-12;

        /// <summary>
        /// Contains a zero-sparse representation of the distribution over labels.
        /// </summary>
        private readonly Dictionary<int, double> probabilities;

        /// <summary>
        /// Initializes a new instance of the <see cref="LabelDistribution"/> class.
        /// </summary>
        public LabelDistribution()
        {
            this.LabelSet = new IndexedSet<string>();
            this.probabilities = new Dictionary<int, double>();
        }

        #endregion

        #region Properties and methods

        /// <summary>
        /// Gets a value indicating whether or not this label distribution is a point mass.
        /// </summary>
        public bool IsPointMass
        {
            get
            {
                if (this.LabelSet.Count <= 1)
                {
                    return true;
                }

                // Uniform distribution
                if (this.probabilities.Count == 0)
                {
                    return false;
                }

                // If any element is not equal to 0 or 1 then this is not a point mass as probabilities are always normalized
                return !this.probabilities.Values.Any(p => 0 < p && p < 1);
            }
        }

        /// <summary>
        /// Gets or sets the bidirectional mapping from labels to label indexes.
        /// </summary>
        public IndexedSet<string> LabelSet { get; set; }

        /// <summary>
        /// Parses a specified string for a label distribution.
        /// </summary>
        /// <param name="specification">The string to be parsed for a label distribution.</param>
        /// <param name="labelSet">A bidirectional mapping from labels to label indexes.</param>
        /// <returns>A label distribution that is equivalent to the given string <see cref="specification"/>.</returns>
        public static LabelDistribution Parse(string specification, IndexedSet<string> labelSet = null)
        {
            string unparsedSpecification;
            return Parse(specification, labelSet, out unparsedSpecification);
        }

        /// <summary>
        /// Parses a specified string for a label distribution.
        /// </summary>
        /// <param name="specification">The string to be parsed for a label distribution.</param>
        /// <param name="labelSet">A bidirectional mapping from labels to label indexes.</param>
        /// <param name="unparsedSpecification">The unparsed part of <see cref="specification"/>.</param>
        /// <returns>A label distribution that is equivalent to the given string <see cref="specification"/>.</returns>
        public static LabelDistribution Parse(string specification, IndexedSet<string> labelSet, out string unparsedSpecification)
        {
            var labelDistribution = new LabelDistribution();
            if (labelSet != null)
            {
                labelDistribution.LabelSet = labelSet;
            }

            unparsedSpecification = labelDistribution.ParseLabelDistribution(specification);

            return labelDistribution;
        }

        /// <summary>
        /// Gets the mode of the distribution.
        /// </summary>
        /// <returns>The mode of the distribution.</returns>
        public string GetMode()
        {
            var distribution = this.ToDictionary();
            double max = double.NegativeInfinity;
            string maxLabel = null;
            foreach (var probability in distribution)
            {
                if (probability.Value > max)
                {
                    max = probability.Value;
                    maxLabel = probability.Key;
                }
            }

            return maxLabel;
        }

        /// <summary>
        /// Returns the label distribution as a dictionary over labels.
        /// </summary>
        /// <returns>The distribution over labels as a dictionary.</returns>
        public IDictionary<string, double> ToDictionary()
        {
            // Probability mass is spread uniformly across unspecified labels
            int unspecifiedCount = this.LabelSet.Count - this.probabilities.Count;
            Debug.Assert(unspecifiedCount >= 0, "More probabilities than labels.");

            double unspecifiedValue = 0.0;
            double sum = this.probabilities.Values.Sum();
            if (sum < 1 && unspecifiedCount > 0)
            {
                unspecifiedValue = (1 - sum) / unspecifiedCount;
            }
            
            IDictionary<string, double> distribution = new Dictionary<string, double>();
            foreach (var labelIndex in this.LabelSet.Indexes)
            {
                double probability;
                if (!this.probabilities.TryGetValue(labelIndex, out probability))
                {
                    probability = unspecifiedValue;
                }
                else
                {
                    if (sum > 1 || (sum < 1 && unspecifiedCount == 0))
                    {
                        probability /= sum;
                    }
                }

                distribution.Add(this.LabelSet.GetElementByIndex(labelIndex), probability);
            }

            Debug.Assert(Math.Abs(distribution.Values.Sum() - 1) < Tolerance, "The label distribution does not satisfy the sum constraint.");

            return distribution;
        }

        /// <summary>
        /// Returns the label distribution as a string.
        /// </summary>
        /// <returns>The label distribution as a string.</returns>
        public override string ToString()
        {
            if (this.IsPointMass)
            {
                return this.GetMode();
            }

            var builder = new StringBuilder();
            var distribution = this.ToDictionary();
            foreach (var point in distribution)
            {
                builder.AppendFormat("{0}{1}{2}{3}", PointSeparator, point.Key, LabelProbabilitySeparator, point.Value);
            }

            return builder.ToString();
        }

        #endregion

        #region Helper methods

        /// <summary>
        /// Parses a specified string for a label distribution.
        /// </summary>
        /// <param name="specification">The string to be parsed for a label distribution.</param>
        /// <returns>The unparsed part of <paramref name="specification"/>.</returns>
        protected string ParseLabelDistribution(string specification)
        {
            specification = specification.Trim();
            
            if (string.IsNullOrEmpty(specification))
            {
                // No specification allowed
                return specification;
            }

            string[] splits = specification.Split(PointSeparator);
            var trimmedSplit = splits[0].Trim();
            if (string.IsNullOrEmpty(trimmedSplit))
            {
                // No specification allowed
                return specification;
            }

            if (trimmedSplit.Contains(":"))
            {
                // A colon is illegal in a label description.
                return specification;
            }

            int i = 0;
            var uncertainLabel = splits[i].Split(LabelProbabilitySeparator).ToList();
            uncertainLabel.RemoveAll(entry => entry == null || entry.Trim().Equals(string.Empty));

            if (uncertainLabel.Count == 1)
            {
                this.ParseLabelProbability(uncertainLabel[0]);
                i++;
            }

            while (uncertainLabel.Count == 2)
            {
                this.ParseLabelProbability(uncertainLabel[0], uncertainLabel[1]);

                if (++i >= splits.Length)
                {
                    break;
                }
                
                uncertainLabel = splits[i].Split(LabelProbabilitySeparator).ToList();
                uncertainLabel.RemoveAll(entry => entry == null || entry.Trim().Equals(string.Empty));
            }

            return this.AssembleUnparsed(splits, i);
        }

        /// <summary>
        /// Parses the specified label and probability strings.
        /// </summary>
        /// <param name="label">The label string to parse.</param>
        /// <param name="probability">The probability string to parse.</param>
        private void ParseLabelProbability(string label, string probability)
        {
            this.ParseLabelProbability(label, this.ParseProbability(probability));
        }

        /// <summary>
        /// Parses a specified label string given its probability.
        /// </summary>
        /// <param name="label">The label string to parse.</param>
        /// <param name="probability">The probability. Defaults to 1.</param>
        private void ParseLabelProbability(string label, double probability = 1.0)
        {
            label = this.ParseLabel(label);
            
            int labelIndex = this.LabelSet.Add(label, false);

            if (this.probabilities.ContainsKey(labelIndex))
            {
                throw new InvalidFileFormatException("The class label '" + label + "' must not be specified multiple times.");
            }
            
            if (probability > 0.0)
            {
                this.probabilities.Add(labelIndex, probability);
            }
        }

        /// <summary>
        /// Parses a specified label string.
        /// </summary>
        /// <param name="label">The label string to parse.</param>
        /// <returns>The parsed label.</returns>
        private string ParseLabel(string label)
        {
            string parsedLabel = label.Trim();
            if (string.IsNullOrEmpty(parsedLabel))
            {
                throw new InvalidFileFormatException("The class label must not be null, empty or whitespace.");
            }

            return parsedLabel;
        }

        /// <summary>
        /// Parses a specified probability string.
        /// </summary>
        /// <param name="probability">The probability string to parse.</param>
        /// <returns>The parsed probability.</returns>
        private double ParseProbability(string probability)
        {
            double parsedProbability;
            if (!double.TryParse(probability.Trim(), out parsedProbability))
            {
                throw new InvalidFileFormatException("Unable to parse class label probability '" + probability.Trim() + "'."); 
            }

            return parsedProbability;
        }

        /// <summary>
        /// Assembles unparsed strings.
        /// </summary>
        /// <param name="splits">The parsed and unparsed strings.</param>
        /// <param name="start">The start index of the unparsed strings.</param>
        /// <returns>The assembled unparsed strings.</returns>
        private string AssembleUnparsed(string[] splits, int start = 1)
        {
            var builder = new StringBuilder();
            for (int i = start; i < splits.Length; i++)
            {
                builder.Append(splits[i] + PointSeparator);
            }

            return builder.ToString();
        }

        #endregion
    }
}
