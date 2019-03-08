// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// The formatting settings used by the <see cref="RegexpAutomatonFormat"/> class.
    /// </summary>
    public class RegexpFormattingSettings
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="RegexpFormattingSettings"/> class.
        /// </summary>
        /// <param name="putOptionalInSquareBrackets">
        /// Whether optional strings should be represented as <c>'[str]'</c> instead of <c>'(str|)'</c>.
        /// </param>
        /// <param name="showAnyElementAsQuestionMark">
        /// Whether languages consisting of a single arbitrary character should be represented
        /// by '?' instead of '.'.
        /// </param>
        /// <param name="ignoreElementDistributionDetails">
        /// Whether non point mass element distributions should be presented to a user as uniform.
        /// </param>
        /// <param name="truncationLength">
        /// The string length to truncate at, for large automata.
        /// </param>
        /// <param name="escapeCharacters">This tells, whether or not characters are to be escaped to be compatible with regex engines.</param>
        /// <param name="useLazyQuantifier">Whether to use lazy quantifiers.</param>
        public RegexpFormattingSettings(
            bool putOptionalInSquareBrackets,
            bool showAnyElementAsQuestionMark,
            bool ignoreElementDistributionDetails,
            int truncationLength,
            bool escapeCharacters,
            bool useLazyQuantifier)
        {
            this.PutOptionalInSquareBrackets = putOptionalInSquareBrackets;
            this.ShowAnyElementAsQuestionMark = showAnyElementAsQuestionMark;
            this.IgnoreElementDistributionDetails = ignoreElementDistributionDetails;
            this.TruncationLength = truncationLength;
            this.EscapeCharacters = escapeCharacters;
            this.UseLazyQuantifier = useLazyQuantifier;
        }

        /// <summary>
        /// Gets a value indicating whether optional strings should be represented as <c>'[str]'</c> instead of <c>'(str|)'</c>.
        /// </summary>
        public bool PutOptionalInSquareBrackets { get; private set; }

        /// <summary>
        /// Gets a value indicating whether languages consisting of a single arbitrary character should be represented
        /// by '?' instead of '.'.
        /// </summary>
        public bool ShowAnyElementAsQuestionMark { get; private set; }

        /// <summary>
        /// Gets a value indicating whether non point mass element distributions
        /// should be presented to a user in a detailed form.
        /// </summary>
        public bool IgnoreElementDistributionDetails { get; private set; }

        /// <summary>
        /// Gets a value for the length of the formatted string at which truncation should occur.
        /// </summary>
        /// <remarks>
        /// This setting keeps the string form of large automata manageable by
        /// truncating it at a certain length.
        /// </remarks>
        public int TruncationLength { get; private set; }

        /// <summary>
        /// Gets a value indicating whether or not characters are to be escaped to be compatible with regex engines.
        /// </summary>
        public bool EscapeCharacters { get; private set; }

        /// <summary>
        /// Gets a value indicating whether to use lazy quantifiers.
        /// </summary>
        public bool UseLazyQuantifier
        {
            get;
            private set;
        }
        
        /// <summary>
        /// Compares this object with a given one.
        /// </summary>
        /// <param name="obj">The object to compare with.</param>
        /// <returns>
        /// <see langword="true"/> if this object equals to the given one, <see langword="false"/> otherwise.
        /// </returns>
        public override bool Equals(object obj)
        {
            if (obj == null || GetType() != obj.GetType())
            {
                return false;
            }

            var objCasted = (RegexpFormattingSettings)obj;
            return
                objCasted.PutOptionalInSquareBrackets == this.PutOptionalInSquareBrackets &&
                objCasted.ShowAnyElementAsQuestionMark == this.ShowAnyElementAsQuestionMark &&
                objCasted.IgnoreElementDistributionDetails == this.IgnoreElementDistributionDetails &&
                objCasted.TruncationLength == this.TruncationLength &&
                objCasted.EscapeCharacters == this.EscapeCharacters &&
                objCasted.UseLazyQuantifier == this.UseLazyQuantifier;
        }

        /// <summary>
        /// Computes the hash code of this object.
        /// </summary>
        /// <returns>
        /// The computed hash code.
        /// </returns>
        public override int GetHashCode()
        {
            int hash = this.PutOptionalInSquareBrackets.GetHashCode();
            hash = Hash.Combine(hash, this.ShowAnyElementAsQuestionMark.GetHashCode());
            hash = Hash.Combine(hash, this.IgnoreElementDistributionDetails.GetHashCode());
            hash = Hash.Combine(hash, this.TruncationLength.GetHashCode());
            hash = Hash.Combine(hash, this.EscapeCharacters.GetHashCode());
            hash = Hash.Combine(hash, this.UseLazyQuantifier.GetHashCode());
            return hash;
        }
    }
}