// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Collections.Generic;
    using System.Collections.Specialized;
    using System.Linq;
    using System.Runtime.Serialization;
    using System.Text;
    using System.Text.RegularExpressions;

    using Collections;
    using Math;
    using Utilities;
    using Factors.Attributes;
    using Serialization;

    /// <summary>
    /// Represents a distribution over characters.
    /// </summary>
    /// <remarks>
    /// This is an optimized version of an earlier version of this class.
    /// It doesn't use <see cref="PiecewiseVector"/> and <see cref="GenericDiscreteBase{T, TThis}"/>,
    /// tries to avoid non-inlineable calls at all costs,
    /// and assumes that the number of constant-probability character ranges is small (1-10).
    /// </remarks>
    [Quality(QualityBand.Experimental)]
    [Serializable]
    [DataContract]
    public sealed class DiscreteChar
        : IDistribution<char>, SettableTo<DiscreteChar>, SettableToProduct<DiscreteChar>, SettableToRatio<DiscreteChar>, SettableToPower<DiscreteChar>,
        SettableToWeightedSumExact<DiscreteChar>, SettableToPartialUniform<DiscreteChar>,
        CanGetLogAverageOf<DiscreteChar>, CanGetLogAverageOfPower<DiscreteChar>, CanGetAverageLog<DiscreteChar>, CanGetMode<char>,
        Sampleable<char>, CanEnumerateSupport<char>
    {
        #region Constants

        /// <summary>
        /// Ranges of only lowercase characters, excluding the mixed range.
        /// </summary>
        public const string OnlyLowerCaseCharacterRanges = "azßöøÿ";

        /// <summary>
        /// Ranges of only uppercase characters, excluding the mixed range
        /// </summary>
        public const string OnlyUpperCaseCharacterRanges = "AZÀÖØÞ";

        /// <summary>
        /// Ranges of mixed upper and lower case characters, which are merged for computational efficiency reasons.
        /// </summary>
        /// <remarks>
        /// In some Unicode ranges,  upper and lower characters alternate which makes it very expensive to 
        /// represent just lower or just upper characters using ranges.  Instead, we use a loose approximation
        /// and allow characters in such ranges to be considered either lower or upper case.
        /// 
        /// This approximation means that upper and lower case character ranges are no longer mutually exclusive.
        /// </remarks>
        public const string MixedCaseCharacterRanges = ""; // disabling any mixed characters for now "Āƿ";

        /// <summary>
        /// Ranges of lowercase characters (for <see cref="Lower"/>).
        /// </summary>
        public const string LowerCaseCharacterRanges = OnlyLowerCaseCharacterRanges + MixedCaseCharacterRanges;


        /// <summary>
        /// Ranges of uppercase characters (for <see cref="Upper"/>).
        /// </summary>
        public const string UpperCaseCharacterRanges = OnlyUpperCaseCharacterRanges + MixedCaseCharacterRanges;

        /// <summary>
        /// Ranges of letters (for <see cref="Letter"/>).
        /// </summary>
        public const string LetterCharacterRanges = OnlyUpperCaseCharacterRanges + OnlyLowerCaseCharacterRanges + MixedCaseCharacterRanges;

        /// <summary>
        /// The tolerance value for probability comparisons.
        /// </summary>
        private const double Eps = 1e-15;

        /// <summary>
        /// The (exclusive) end of the character range.
        /// </summary>
        private const int CharRangeEndExclusive = char.MaxValue + 1;

        /// <summary>
        /// The probability of a character under a uniform distribution over characters.
        /// </summary>
        private const double UniformProb = 1.0 / CharRangeEndExclusive;

        /// <summary>
        /// The index value used by <see cref="GetRangeIndexForCharacter"/> to indicate that no range has been found.
        /// </summary>
        private const int UnknownRange = -1;

        /// <summary>
        /// The initial capacity of <see cref="ranges"/>.
        /// </summary>
        private const int InitialRangeArrayCapacity = 3;

        private const string DigitRegexRepresentation = @"\d";
        private const string DigitSymbolRepresentation = @"#";

        private const string LetterRegexRepresentation = @"[\p{Ll}\p{Lu}]";
        private const string LetterSymbolRepresentation = @"↕";

        private const string LetterOrDigitRegexRepresentation = @"[\p{Ll}\p{Lu}\d]";
        private const string LetterOrDigitSymbolRepresentation = @"◊";

        private const string LowerRegexRepresentation = @"\p{Ll}";
        private const string LowerSymbolRepresentation = @"↓";

        private const string UpperRegexRepresentation = @"\p{Lu}";
        private const string UpperSymbolRepresentation = @"↑";

        private const string WordCharRegexRepresentation = @"\w";
        private const string WordCharSymbolRepresentation = @"○";

        private const string UniformRegexRepresentation = ".";
        private const string UniformSymbolRepresentation = ".";

        #endregion

        #region State

        //// Don't forget to modify SetTo(), Swap() and Clone() after editing this section.

        /// <summary>
        /// The array of character ranges with associated probabilities.
        /// </summary>
        /// <remarks>
        /// The character probabilities must be kept normalized by applying <see cref="Normalize"/> when necessary.
        /// </remarks>
        [DataMember]
        private CharRange[] ranges;

        /// <summary>
        /// The number of actually used values in <see cref="ranges"/>.
        /// </summary>
        /// <remarks>
        /// We don't use <see cref="List{T}"/> for <see cref="ranges"/> for performance reasons.
        /// </remarks>
        [DataMember]
        private int rangeCount;

        /// <summary>
        /// The probability of a character outside character ranges defined by <see cref="ranges"/>.
        /// </summary>
        /// <remarks>
        /// The character probabilities must be kept normalized by applying <see cref="Normalize"/> when necessary.
        /// </remarks>
        [DataMember]
        private double probabilityOutsideRanges;

        [DataMember]
        private CharClasses charClasses = CharClasses.Unknown;

        private string regexRepresentation = null;
        private string symbolRepresentation = null;
        #endregion

        /// <summary>
        /// Initializes a new instance of the <see cref="DiscreteChar"/> class
        /// by setting it to a uniform distribution.
        /// </summary>
        public DiscreteChar()
        {
            this.ranges = new CharRange[InitialRangeArrayCapacity];
            this.probabilityOutsideRanges = UniformProb;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="DiscreteChar"/> class
        /// with a given list of constant probability character ranges and the probability of characters outside ranges.
        /// </summary>
        /// <param name="probabilityOutsideRanges">The probability of characters outside the given ranges.</param>
        /// <param name="ranges">The constant-probability character ranges.</param>
        /// <param name="rangeCount">The number of valid elements in the <paramref name="ranges"/> array.</param>
        /// <remarks>
        /// The probabilities need to be normalized. The character ranges need to be sorted.
        /// The created objects takes ownership of the character range list.
        /// </remarks>
        private DiscreteChar(double probabilityOutsideRanges, CharRange[] ranges, int rangeCount)
        {
            this.probabilityOutsideRanges = probabilityOutsideRanges;
            this.ranges = ranges;
            this.rangeCount = rangeCount;
        }

        #region Properties

        #region Properties matching factory methods

        /// <summary>
        /// Gets a value indicating whether this distribution equals the distribution created by <see cref="Digit"/>.
        /// </summary>
        public bool IsDigit
        {
            // TODO: there exists more than one representation of this distribution
            get
            {
                if (this.charClasses == CharClasses.Unknown && this.Equals(Digit()))
                {
                    this.charClasses = CharClasses.Digit;
                }

                return this.charClasses == CharClasses.Digit;
            }
        }

        /// <summary>
        /// Gets a value indicating whether this distribution equals the distribution created by <see cref="Lower"/>.
        /// </summary>
        public bool IsLower
        {
            // TODO: there exists more than one representation of this distribution
            get
            {
                if (this.charClasses == CharClasses.Unknown && this.Equals(Lower()))
                {
                    this.charClasses = CharClasses.Lower;
                }

                return this.charClasses == CharClasses.Lower;
            }
        }

        /// <summary>
        /// Gets a value indicating whether this distribution equals the distribution created by <see cref="Upper"/>.
        /// </summary>
        public bool IsUpper
        {
            // TODO: there exists more than one representation of this distribution
            get
            {
                if (this.charClasses == CharClasses.Unknown && this.Equals(Upper()))
                {
                    this.charClasses = CharClasses.Upper;
                }

                return this.charClasses == CharClasses.Upper;
            }
        }

        /// <summary>
        /// Gets a value indicating whether this distribution equals the distribution created by <see cref="Letter"/>.
        /// </summary>
        public bool IsLetter
        {
            // TODO: there exists more than one representation of this distribution
            get
            {
                if (this.charClasses == CharClasses.Unknown && this.Equals(Letter()))
                {
                    this.charClasses = CharClasses.Letter;
                }

                return this.charClasses == CharClasses.Letter;
            }
        }

        /// <summary>
        /// Gets a value indicating whether this distribution equals the distribution created by <see cref="LetterOrDigit"/>.
        /// </summary>
        public bool IsLetterOrDigit
        {
            // TODO: there exists more than one representation of this distribution
            get
            {
                if (this.charClasses == CharClasses.Unknown && this.Equals(LetterOrDigit()))
                {
                    this.charClasses = CharClasses.LetterOrDigit;
                }

                return this.charClasses == CharClasses.LetterOrDigit;
            }
        }

        /// <summary>
        /// Gets a value indicating whether this distribution equals the distribution created by <see cref="WordChar"/>.
        /// </summary>
        public bool IsWordChar
        {
            // TODO: there exists more than one representation of this distribution
            get
            {
                if (this.charClasses == CharClasses.Unknown && this.Equals(WordChar()))
                {
                    this.charClasses = CharClasses.WordChar;
                }

                return this.charClasses == CharClasses.WordChar;
            }
        }

        #endregion

        #region Distribution properties

        /// <summary>
        /// Gets the probability assigned to characters outside ranges returned by <see cref="GetRanges"/>.
        /// </summary>
        public double ProbabilityOutsideRanges
        {
            get { return this.probabilityOutsideRanges; }
        }

        /// <summary>
        /// Gets or sets the point mass represented by the distribution.
        /// </summary>
        public char Point
        {
            get
            {
                if (!this.IsPointMass)
                {
                    throw new InvalidOperationException();
                }

                // TODO: see comments in IsPointMass
                return (char)this.ranges[0].StartInclusive;
            }

            set
            {
                this.probabilityOutsideRanges = 0;
                this.rangeCount = 0;
                this.AddRange(new CharRange { StartInclusive = value, EndExclusive = value + 1, Probability = 1 });
            }
        }

        /// <summary>
        /// Gets a value indicating whether this distribution represents a point mass.
        /// </summary>
        public bool IsPointMass
        {
            get
            {
                // TODO: Assumes that there are no ranges with zero probability
                // TODO: also assumes that a point is not represented by zero-probability ranges and a non-zero value outside of ranges
                return this.rangeCount > 0 && Math.Abs(this.ranges[0].Probability - 1) < Eps;
            }
        }

        /// <summary>
        /// Gets the probability of a given character under this distribution.
        /// </summary>
        /// <param name="value">The character.</param>
        /// <returns>The probability of the character under this distribution.</returns>
        public double this[char value]
        {
            get
            {
                int index = this.GetRangeIndexForCharacter(value);
                return index == UnknownRange ? this.probabilityOutsideRanges : this.ranges[index].Probability;
            }
        }

        #endregion

        #endregion

        #region Factory methods

        /// <summary>
        /// Creates a distribution given a list of constant probability character ranges and the probability of characters outside those ranges.
        /// </summary>
        /// <param name="probabilityOutsideRanges">The probability of characters outside the given ranges.</param>
        /// <param name="ranges">The constant-probability character ranges.</param>
        /// <remarks>The probabilities do not need to be normalized. The character ranges do not need to be sorted.</remarks>
        /// <returns>The created distribution.</returns>
        [Construction("ProbabilityOutsideRanges", "GetRanges")]
        public static DiscreteChar Create(double probabilityOutsideRanges, IEnumerable<CharRange> ranges)
        {
            var result = new DiscreteChar();
            
            Argument.CheckIfNotNull(ranges, "ranges");
            result.ranges = ranges.ToArray();
            result.rangeCount = result.ranges.Length;

            Array.Sort(result.ranges, (s1, s2) => Comparer<int>.Default.Compare(s1.StartInclusive, s2.StartInclusive));
            CheckRanges(result.ranges, "ranges");

            CheckUnnormalizedProbability(probabilityOutsideRanges, "probabilityOutsideRanges");
            result.probabilityOutsideRanges = probabilityOutsideRanges;

            result.Normalize();
            result.MergeNeighboringRanges();

            return result;
        }
        
        /// <summary>
        /// Creates a uniform distribution over characters.
        /// </summary>
        /// <returns>The created distribution.</returns>
        [Construction(UseWhen = "IsUniform")]
        public static DiscreteChar Uniform()
        {
            var c = new DiscreteChar();
            c.charClasses = CharClasses.Uniform;
            c.regexRepresentation = UniformRegexRepresentation;
            return c;
        }

        /// <summary>
        /// Creates a uniform distribution over digits '0'..'9'.
        /// </summary>
        /// <returns>The created distribution.</returns>
        [Construction(UseWhen = "IsDigit")]
        public static DiscreteChar Digit()
        {
            var c = DiscreteChar.InRange('0', '9');
            c.charClasses = CharClasses.Digit;
            c.regexRepresentation = DigitRegexRepresentation;
            return c;
        }

        /// <summary>
        /// Creates a uniform distribution over lowercase letters.
        /// </summary>
        /// <returns>The created distribution.</returns>
        [Construction(UseWhen = "IsLower")]
        public static DiscreteChar Lower()
        {
            var c = DiscreteChar.UniformInRanges(LowerCaseCharacterRanges);
            c.charClasses = CharClasses.Lower;
            c.regexRepresentation = LowerRegexRepresentation;
            return c;
        }

        /// <summary>
        /// Creates a uniform distribution over uppercase letters.
        /// </summary>
        /// <returns>The created distribution.</returns>
        [Construction(UseWhen = "IsUpper")]
        public static DiscreteChar Upper()
        {
            var c = DiscreteChar.UniformInRanges(UpperCaseCharacterRanges);
            c.charClasses = CharClasses.Upper;
            c.regexRepresentation = UpperRegexRepresentation;
            return c;
        }

        /// <summary>
        /// Creates a uniform distribution over letters.
        /// </summary>
        /// <returns>The created distribution.</returns>
        public static DiscreteChar Letter()
        {
            var c = DiscreteChar.UniformInRanges(LetterCharacterRanges);
            c.charClasses = CharClasses.Letter;
            c.regexRepresentation = LetterRegexRepresentation;
            return c;
        }

        /// <summary>
        /// Creates a uniform distribution over letters and '0'..'9'.
        /// </summary>
        /// <returns>The created distribution.</returns>
        public static DiscreteChar LetterOrDigit()
        {
            var c = DiscreteChar.UniformInRanges(LetterCharacterRanges + "09");
            c.charClasses = CharClasses.LetterOrDigit;
            c.regexRepresentation = LetterOrDigitRegexRepresentation;
            return c;
        }

        /// <summary>
        /// Creates a uniform distribution over word characters (letter, digit and '_').
        /// </summary>
        /// <returns>The created distribution.</returns>
        public static DiscreteChar WordChar()
        {
            var c = DiscreteChar.UniformInRanges(LetterCharacterRanges + "09__");
            c.charClasses = CharClasses.WordChar;
            c.regexRepresentation = WordCharRegexRepresentation;
            return c;
        }

        /// <summary>
        /// Creates a uniform distribution over all characters except (letter, digit and '_').
        /// </summary>
        /// <returns>The created distribution.</returns>
        public static DiscreteChar NonWordChar()
        {
            return DiscreteChar.WordChar().Complement();
        }

        /// <summary>
        /// Creates a uniform distribution over whitespace characters ('\t'..'\r', ' ').
        /// </summary>
        /// <returns>The created distribution.</returns>
        public static DiscreteChar Whitespace()
        {
            return DiscreteChar.InRanges("\t\r  ");
        }

        /// <summary>
        /// Creates a uniform distribution over all characters.
        /// </summary>
        /// <returns>The created distribution.</returns>
        public static DiscreteChar Any()
        {
            return DiscreteChar.Uniform();
        }

        /// <summary>
        /// Creates a uniform distribution over characters in a given range.
        /// </summary>
        /// <param name="start">The start of the range (inclusive).</param>
        /// <param name="end">The end of the range (inclusive).</param>
        /// <returns>The created distribution.</returns>
        public static DiscreteChar InRange(char start, char end)
        {
            return DiscreteChar.UniformInRanges(start, end);
        }

        /// <summary>
        /// Creates a distribution which is uniform over values in
        /// multiple ranges specified by pairs of start and end values. These pairs
        /// are specified as adjacent values in an array whose length must therefore be even.
        /// </summary>
        /// <param name="startEndPairs">The array of range starts and ends.</param>
        /// <returns>The created distribution.</returns>
        public static DiscreteChar InRanges(params char[] startEndPairs)
        {
            return DiscreteChar.UniformInRanges(startEndPairs);
        }

        /// <summary>
        /// Creates a distribution which is uniform over values in
        /// multiple ranges specified by pairs of start and end values. These pairs
        /// are specified as adjacent values in a sequence whose length must therefore be even.
        /// </summary>
        /// <param name="startEndPairs">The sequence of range starts and ends.</param>
        /// <returns>The created distribution.</returns>
        public static DiscreteChar InRanges(IEnumerable<char> startEndPairs)
        {
            return DiscreteChar.UniformInRanges(startEndPairs);
        }

        /// <summary>
        /// Creates a distribution which is uniform over the specified set of characters.
        /// </summary>
        /// <param name="chars">The characters.</param>
        /// <returns>The created distribution.</returns>
        public static DiscreteChar OneOf(params char[] chars)
        {
            return DiscreteChar.UniformOver(chars);
        }

        /// <summary>
        /// Creates a distribution which is uniform over the specified set of characters.
        /// </summary>
        /// <param name="chars">The characters.</param>
        /// <returns>The created distribution.</returns>
        public static DiscreteChar OneOf(IEnumerable<char> chars)
        {
            return DiscreteChar.UniformOver(chars);
        }

        /// <summary>
        /// Creates a distribution which is uniform over the specified set of characters.
        /// </summary>
        /// <param name="chars">The characters.</param>
        /// <returns>The created distribution.</returns>
        public static DiscreteChar UniformOver(params char[] chars)
        {
            return UniformOver((IEnumerable<char>)chars);
        }

        /// <summary>
        /// Creates a distribution which is uniform over the specified set of characters.
        /// </summary>
        /// <param name="chars">The characters.</param>
        /// <returns>The created distribution.</returns>
        public static DiscreteChar UniformOver(IEnumerable<char> chars)
        {
            Argument.CheckIfNotNull(chars, "chars");
            
            var rangeBounds = new List<char>();
            foreach (char c in chars)
            {
                rangeBounds.Add(c);
                rangeBounds.Add(c);
            }

            return UniformInRanges(rangeBounds);
        }

        /// <summary>
        /// Creates a uniform distribution over characters in a given range.
        /// </summary>
        /// <param name="start">The start of the range (inclusive).</param>
        /// <param name="end">The end of the range (inclusive).</param>
        /// <returns>The created distribution.</returns>
        public static DiscreteChar UniformInRange(char start, char end)
        {
            return UniformInRanges(start, end);
        }

        /// <summary>
        /// Creates a distribution which is uniform over values in
        /// multiple ranges specified by pairs of start and end values. These pairs
        /// are specified as adjacent values in an array whose length must therefore be even.
        /// </summary>
        /// <param name="startEndPairs">The array of range starts and ends.</param>
        /// <returns>The created distribution.</returns>
        public static DiscreteChar UniformInRanges(params char[] startEndPairs)
        {
            return UniformInRanges((IEnumerable<char>)startEndPairs);
        }

        /// <summary>
        /// Creates a distribution which is uniform over values in
        /// multiple ranges specified by pairs of start and end values. These pairs
        /// are specified as adjacent values in a sequence whose length must therefore be even.
        /// </summary>
        /// <param name="startEndPairs">The sequence of range starts and ends.</param>
        /// <returns>The created distribution.</returns>
        public static DiscreteChar UniformInRanges(IEnumerable<char> startEndPairs)
        {
            Argument.CheckIfNotNull(startEndPairs, "startEndPairs");
            var startEndPairsArray = startEndPairs.ToArray();
            Argument.CheckIfValid(startEndPairsArray.Length % 2 == 0, "startEndPairs", "The number of characters must be even.");

            var ranges = new List<CharRange>();
            for (int i = 0; i < startEndPairsArray.Length; i += 2)
            {
                var startInclusive = startEndPairsArray[i];
                var endExclusive = startEndPairsArray[i + 1] + 1;
                if (startInclusive >= endExclusive)
                    throw new ArgumentException("Inverted character range");

                ranges.Add(new CharRange { StartInclusive = startInclusive, EndExclusive = endExclusive, Probability = 1 });
            }

            return DiscreteChar.Create(0, ranges);
        }

        /// <summary>
        /// Creates a point mass character distribution.
        /// </summary>
        /// <param name="point">The point.</param>
        /// <returns>The created distribution.</returns>
        public static DiscreteChar PointMass(char point)
        {
            return new DiscreteChar(0, new[] { new CharRange { StartInclusive = point, EndExclusive = point + 1, Probability = 1 } }, 1);
        }

        /// <summary>
        /// Creates a point character from a vector of (unnormalized) character probabilities.
        /// </summary>
        /// <param name="vector">The vector of unnormalized probabilities.</param>
        /// <returns>The created distribution.</returns>
        public static DiscreteChar FromVector(Vector vector)
        {
            Argument.CheckIfNotNull(vector, "vector");
            
            var piecewise = vector as PiecewiseVector;
            if (piecewise != null)
            {
                return FromVector(piecewise);
            }

            return FromVector(PiecewiseVector.Copy(vector));
        }

        /// <summary>
        /// Creates a point character from a vector of (unnormalized) character probabilities.
        /// </summary>
        /// <param name="vector">The vector of unnormalized probabilities.</param>
        /// <returns>The created distribution.</returns>
        public static DiscreteChar FromVector(PiecewiseVector vector)
        {
            Argument.CheckIfNotNull(vector, "vector");
            Argument.CheckIfValid(vector.Count <= CharRangeEndExclusive, "The given vector is of invalid size.");

            var result = new DiscreteChar();
            result.probabilityOutsideRanges = vector.CommonValue;
            for (int i = 0; i < vector.Pieces.Count; ++i)
            {
                var piece = vector.Pieces[i];
                result.AddRange(new CharRange { StartInclusive = piece.Start, EndExclusive = piece.End + 1, Probability = piece.Value });
            }

            if (vector.Count < CharRangeEndExclusive && Math.Abs(vector.CommonValue) > Eps)
            {
                result.AddRange(new CharRange { StartInclusive = vector.Count, EndExclusive = CharRangeEndExclusive, Probability = 0 });
            }

            result.Normalize();
                        
            return result;
        }
        
        #endregion

        #region Distribution interfaces implementation

        /// <summary>
        /// Creates a copy of this distribution.
        /// </summary>
        /// <returns>The created copy.</returns>
        object ICloneable.Clone()
        {
            return this.Clone();
        }

        /// <summary>
        /// Creates a copy of this distribution.
        /// </summary>
        /// <returns>The created copy.</returns>
        public DiscreteChar Clone()
        {
            return new DiscreteChar(this.probabilityOutsideRanges, (CharRange[])this.ranges.Clone(), this.rangeCount);
        }

        /// <summary>
        /// Gets the maximum difference between the character probabilities under this distribution and a given one.
        /// </summary>
        /// <param name="distribution">The distribution to compute the maximum probability difference with.</param>
        /// <returns>The computed maximum probability difference.</returns>
        public double MaxDiff(object distribution)
        {
            Argument.CheckIfNotNull(distribution, "distribution");
            
            var thatDist = distribution as DiscreteChar;
            if (thatDist == null)
            {
                return double.PositiveInfinity;
            }

            double result = 0;
            foreach (var pair in CharRangePair.CombinedRanges(this, thatDist, false))
            {
                result = Math.Max(result, Math.Abs(pair.Probability1 - pair.Probability2));
            }

            return result;
        }

        /// <summary>
        /// Sets this distribution to a uniform distribution over all characters.
        /// </summary>
        public void SetToUniform()
        {
            this.rangeCount = 0;
            this.probabilityOutsideRanges = UniformProb;
        }

        /// <summary>
        /// Checks whether this distribution is a uniform distribution over all characters.
        /// </summary>
        /// <returns>
        /// <see langword="true"/> if this distribution is a uniform distribution over all possible characters,
        /// <see langword="false"/> otherwise.
        /// </returns>
        public bool IsUniform()
        {
            for (int i = 0; i < this.rangeCount; ++i)
            {
                if (Math.Abs(this.ranges[i].Probability - UniformProb) > Eps)
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Gets the logarithm of the probability of a given character under this distribution.
        /// </summary>
        /// <param name="value">The character.</param>
        /// <returns>
        /// <see langword="true"/> if this distribution is a uniform distribution over all possible characters,
        /// <see langword="false"/> otherwise.
        /// </returns>
        public double GetLogProb(char value)
        {
            return Math.Log(this[value]);
        }

        /// <summary>
        /// Sets this distribution to a product of a given pair of distributions.
        /// </summary>
        /// <param name="distribution1">The first distribution.</param>
        /// <param name="distribution2">The second distribution.</param>
        public void SetToProduct(DiscreteChar distribution1, DiscreteChar distribution2)
        {
            Argument.CheckIfNotNull(distribution1, "distribution1");
            Argument.CheckIfNotNull(distribution2, "distribution2");
            
            var result = new DiscreteChar();
            result.probabilityOutsideRanges = distribution1.probabilityOutsideRanges * distribution2.probabilityOutsideRanges;
            foreach (var pair in CharRangePair.CombinedRanges(distribution1, distribution2))
            {
                double probProduct = pair.Probability1 * pair.Probability2;
                if (Math.Abs(probProduct - result.probabilityOutsideRanges) > Eps)
                {
                    result.AddRange(new CharRange { StartInclusive = pair.StartInclusive, EndExclusive = pair.EndExclusive, Probability = probProduct });
                }
            }

            result.Normalize();
            result.MergeNeighboringRanges();
            
            this.SwapWith(result);
        }

        /// <summary>
        /// Sets this distribution to a weighted sum of a given pair of distributions.
        /// </summary>
        /// <param name="weight1">The weight of the first distribution.</param>
        /// <param name="distribution1">The first distribution.</param>
        /// <param name="weight2">The weight of the second distribution.</param>
        /// <param name="distribution2">The second distribution.</param>
        public void SetToSum(double weight1, DiscreteChar distribution1, double weight2, DiscreteChar distribution2)
        {
            Argument.CheckIfNotNull(distribution1, "distribution1");
            Argument.CheckIfNotNull(distribution2, "distribution2");
            
            if (weight1 + weight2 == 0)
            {
                this.SetToUniform();
            }
            else if (weight1 + weight2 < 0)
            {
                throw new ArgumentException("weight1 (" + weight1 + ") + weight2 (" + weight2 + ") < 0");
            }
            else if (double.IsPositiveInfinity(weight1))
            {
                if (double.IsPositiveInfinity(weight2))
                {
                    throw new ArgumentException("both weights are infinity");
                }

                this.SetTo(distribution1);
            }
            else if (double.IsPositiveInfinity(weight2))
            {
                this.SetTo(distribution2);
            }
            else
            {
                // Make the weights add to 1 to avoid small weights causing a zero mass distribution.
                double invW = 1.0 / (weight1 + weight2);
                weight1 *= invW;
                weight2 *= invW;
                var result = new DiscreteChar();
                result.probabilityOutsideRanges =
                    (weight1 * distribution1.probabilityOutsideRanges) + (weight2 * distribution2.probabilityOutsideRanges);
                foreach (var pair in CharRangePair.CombinedRanges(distribution1, distribution2, false))
                {
                    double probSum = (weight1 * pair.Probability1) + (weight2 * pair.Probability2);
                    if (Math.Abs(probSum - result.probabilityOutsideRanges) > Eps)
                    {
                        result.AddRange(new CharRange { StartInclusive = pair.StartInclusive, EndExclusive = pair.EndExclusive, Probability = probSum });
                    }
                }

                result.Normalize();
                result.MergeNeighboringRanges();
                
                this.SwapWith(result);
            }
        }

        /// <summary>
        /// Returns the logarithm of the probability that the current distribution would draw the same sample as a given one.
        /// </summary>
        /// <param name="distribution">The given distribution.</param>
        /// <returns>The logarithm of the probability that distributions would draw the same sample.</returns>
        public double GetLogAverageOf(DiscreteChar distribution)
        {
            Argument.CheckIfNotNull(distribution, "distribution");
            
            double result = 0;
            foreach (var pair in CharRangePair.CombinedRanges(this, distribution))
            {
                result += pair.Probability1 * pair.Probability2 * (pair.EndExclusive - pair.StartInclusive);
            }

            return Math.Log(result);
        }

        /// <summary>
        /// Sets the distribution to be uniform over its support.
        /// </summary>
        public void SetToPartialUniform()
        {
            this.SetToPartialUniformOf(this);
        }

        /// <summary>
        /// Sets the distribution to be uniform over the support of a given distribution.
        /// </summary>
        /// <param name="distribution">The distribution which support will be used to setup the current distribution.</param>
        public void SetToPartialUniformOf(DiscreteChar distribution)
        {
            Argument.CheckIfNotNull(distribution, "distribution");

            var result = new DiscreteChar();
            result.probabilityOutsideRanges = distribution.probabilityOutsideRanges > Eps ? 1 : 0;

            for (int i = 0; i < distribution.rangeCount; ++i)
            {
                var range = distribution.ranges[i];
                range.Probability = range.Probability > Eps ? 1 : 0;
                result.AddRange(range);
            }

            result.Normalize();
            result.MergeNeighboringRanges();
            
            this.SwapWith(result);
        }

        /// <summary>
        /// Checks whether the distribution is uniform over its support.
        /// </summary>
        /// <returns><see langword="true"/> if the distribution is uniform over its support, <see langword="false"/> otherwise.</returns>
        public bool IsPartialUniform()
        {
            double? commonProb = null;
            bool hasCommonValues = false;
            int prevRangeEnd = 0;
            for (int i = 0; i < this.rangeCount; ++i)
            {
                var range = this.ranges[i];
                if (commonProb.HasValue && range.Probability > Eps && Math.Abs(commonProb.Value - range.Probability) > Eps)
                {
                    return false;
                }

                commonProb = range.Probability;
                hasCommonValues |= range.StartInclusive > prevRangeEnd;
                prevRangeEnd = range.EndExclusive;
            }

            hasCommonValues |= prevRangeEnd < CharRangeEndExclusive;

            if (hasCommonValues && commonProb.HasValue && this.probabilityOutsideRanges > Eps &&
                Math.Abs(commonProb.Value - this.probabilityOutsideRanges) > Eps)
            {
                return false;
            }

            return true;
        }

        /// <summary>
        /// Sets the current distribution to the ratio of a given pair of distributions.
        /// </summary>
        /// <param name="numerator">The numerator in the ratio.</param>
        /// <param name="denominator">The denominator in the ratio.</param>
        /// <param name="forceProper">Specifies whether the ratio must be proper.</param>
        public void SetToRatio(DiscreteChar numerator, DiscreteChar denominator, bool forceProper = false)
        {
            Argument.CheckIfNotNull(numerator, "numerator");
            Argument.CheckIfNotNull(denominator, "denominator");
            
            var result = new DiscreteChar();
            result.probabilityOutsideRanges = DivideProb(numerator.probabilityOutsideRanges, denominator.probabilityOutsideRanges);

            foreach (var pair in CharRangePair.CombinedRanges(numerator, denominator))
            {
                double probRatio = DivideProb(pair.Probability1, pair.Probability2);
                if (Math.Abs(probRatio - result.probabilityOutsideRanges) > Eps)
                {
                    result.AddRange(new CharRange { StartInclusive = pair.StartInclusive, EndExclusive = pair.EndExclusive, Probability = probRatio });
                }
            }

            result.Normalize();
            result.MergeNeighboringRanges();

            this.SwapWith(result);
        }

        /// <summary>
        /// Sets the current distribution to a given distribution raised to a given power.
        /// </summary>
        /// <param name="distribution">The distribution to raise to the power.</param>
        /// <param name="power">The power.</param>
        public void SetToPower(DiscreteChar distribution, double power)
        {
            Argument.CheckIfNotNull(distribution, "distribution");
            
            var result = new DiscreteChar();

            bool hasCommonValues = false;
            int prevRangeEnd = 0;
            for (int i = 0; i < distribution.rangeCount; ++i)
            {
                var range = distribution.ranges[i];
                if (range.Probability < Eps && power < 0)
                {
                    throw new DivideByZeroException();
                }

                range.Probability = Math.Pow(range.Probability, power);
                result.AddRange(range);

                hasCommonValues |= range.StartInclusive > prevRangeEnd;
                prevRangeEnd = range.EndExclusive;
            }

            hasCommonValues |= prevRangeEnd < CharRangeEndExclusive;
            if (hasCommonValues)
            {
                if (distribution.probabilityOutsideRanges < Eps && power < 0)
                {
                    throw new DivideByZeroException();
                }

                result.probabilityOutsideRanges = Math.Pow(distribution.probabilityOutsideRanges, power);
            }
            
            result.Normalize();
            result.MergeNeighboringRanges();

            this.SwapWith(result);
        }

        /// <summary>
        /// Computes the log-integral of one distribution times another raised to a power.
        /// </summary>
        /// <param name="distribution">The other distribution</param>
        /// <param name="power">The power.</param>
        /// <returns><c>Math.Log(sum_x this.Evaluate(x) * Math.Pow(distribution.Evaluate(x), power))</c></returns>
        /// <remarks>
        /// This is not the same as <c>GetLogAverageOf(distribution^power)</c> because it includes the normalization constant of that.
        /// </remarks>
        public double GetLogAverageOfPower(DiscreteChar distribution, double power)
        {
            Argument.CheckIfNotNull(distribution, "distribution");
            
            double result = 0;
            foreach (var pair in CharRangePair.CombinedRanges(this, distribution))
            {
                if (pair.Probability2 < Eps && power < 0)
                {
                    throw new DivideByZeroException();
                }

                result += pair.Probability1 * Math.Pow(pair.Probability2, power) * (pair.EndExclusive - pair.StartInclusive);
            }

            return Math.Log(result);
        }

        /// <summary>
        /// Computes the expected logarithm of a given distribution under this distribution.
        /// </summary>
        /// <param name="distribution">The distribution to take the logarithm of.</param>
        /// <returns><c>sum_x this.Evaluate(x)*Math.Log(distribution.Evaluate(x))</c></returns>
        /// <remarks>This is also known as the cross entropy.</remarks>
        public double GetAverageLog(DiscreteChar distribution)
        {
            Argument.CheckIfNotNull(distribution, "distribution");
            
            double result = 0;
            foreach (var pair in CharRangePair.CombinedRanges(this, distribution, true))
            {
                if (pair.Probability2 == 0.0)
                {
                    return double.NegativeInfinity;
                }

                double product = ValueTimesLogValue(pair.Probability1, pair.Probability2);
                result += product * (pair.EndExclusive - pair.StartInclusive);
            }

            return result;
        }

        /// <summary>
        /// Gets the mode of this distribution.
        /// </summary>
        /// <returns>The mode.</returns>
        public char GetMode()
        {
            bool hasCommonValues = false;
            int prevRangeEnd = 0;
            char mode = '\0';
            char charOutOfRanges = '\0';
            double maxProb = 0;
            for (int i = 0; i < this.rangeCount; ++i)
            {
                var range = this.ranges[i];
                if (range.Probability > maxProb)
                {
                    mode = (char)range.StartInclusive;
                    maxProb = range.Probability;
                }

                if (range.StartInclusive > prevRangeEnd)
                {
                    hasCommonValues = true;
                    charOutOfRanges = (char)prevRangeEnd;
                }
                
                prevRangeEnd = range.EndExclusive;
            }

            if (prevRangeEnd < CharRangeEndExclusive)
            {
                hasCommonValues = true;
                charOutOfRanges = (char)prevRangeEnd;
            }

            return hasCommonValues && this.probabilityOutsideRanges > maxProb ? charOutOfRanges : mode;
        }

        /// <summary>
        /// Draws a sample from the distribution.
        /// </summary>
        /// <returns>The drawn sample.</returns>
        public char Sample()
        {
            var sampleProb = Rand.Double();
            
            foreach (var interval in EnumerateCharRanges())
            {
                var intervalLength = interval.EndExclusive - interval.StartInclusive;
                var prob = intervalLength * interval.Probability;
                sampleProb -= prob;
                if (sampleProb < 0)
                {
                    return (char)(interval.StartInclusive - sampleProb / prob * intervalLength);
                }
            }

            throw new Exception();
        }

        private IEnumerable<CharRange> EnumerateCharRanges()
        {
            var prevRangeEnd = 0;
            for (var i = 0; i < rangeCount; i++)
            {
                var range = ranges[i];
                yield return new CharRange(prevRangeEnd, range.StartInclusive, probabilityOutsideRanges);
                yield return new CharRange(range.StartInclusive, range.EndExclusive, range.Probability);
                prevRangeEnd = range.EndExclusive;
            }

            yield return new CharRange(prevRangeEnd, CharRangeEndExclusive, probabilityOutsideRanges);
        }

        /// <summary>
        /// Draws a sample from the distribution.
        /// </summary>
        /// <param name="result">A pre-allocated storage for the sample (will be ignored).</param>
        /// <returns>The drawn sample.</returns>
        public char Sample(char result)
        {
            return this.Sample();
        }

        /// <summary>
        /// Sets this distribution to be equal to a given distribution.
        /// </summary>
        /// <param name="distribution">The distribution to set this distribution to.</param>
        public void SetTo(DiscreteChar distribution)
        {
            Argument.CheckIfNotNull(distribution, "distribution");
            
            if (!ReferenceEquals(this, distribution))
            {
                this.probabilityOutsideRanges = distribution.probabilityOutsideRanges;
                
                this.rangeCount = 0;
                for (int i = 0; i < distribution.rangeCount; ++i)
                {
                    this.AddRange(distribution.ranges[i]);
                }
            }
        }

        /// <summary>
        /// Enumerates over the support of the distribution instance.
        /// </summary>
        /// <returns>The character values with non-zero mass.</returns>
        public IEnumerable<char> EnumerateSupport()
        {
            int prevRangeEnd = 0;

            for (int i = 0; i < this.rangeCount; ++i)
            {
                var range = this.ranges[i];
                if (this.ProbabilityOutsideRanges > 0.0)
                {
                    for (int j = prevRangeEnd; j < range.StartInclusive;  j++)
                    {
                        yield return (char)j;
                    }
                }

                if (range.Probability > 0.0)
                {
                    for (int j = range.StartInclusive; j < range.EndExclusive; j++)
                    {
                        yield return (char)j;
                    }
                }

                prevRangeEnd = range.EndExclusive;
            }

            if (this.ProbabilityOutsideRanges > 0.0)
            {
                for (int j = prevRangeEnd; j < CharRangeEndExclusive; j++)
                {
                    yield return (char)j;
                }
            }
        }
        #endregion

        #region Character distribution specific interface

        /// <summary>
        /// Gets an array of character ranges with associated probabilities.
        /// </summary>
        /// <remarks>
        /// See <see cref="ProbabilityOutsideRanges"/> for the probability of characters not covered by the returned ranges.
        /// </remarks>
        /// <returns>An array of character ranges with associated probabilities.</returns>
        public CharRange[] GetRanges()
        {
            var result = new CharRange[this.rangeCount];
            Array.Copy(this.ranges, result, this.rangeCount);
            return result;
        }
        
        /// <summary>
        /// Creates a distribution which is uniform over all characters
        /// that have zero probability under this distribution
        /// i.e. that are not 'in' this distribution.
        /// </summary>
        /// <remarks>
        /// This is useful for defining characters that are not in a particular distribution
        /// e.g. not a letter or not a word character.
        /// </remarks>
        /// <returns>The created distribution.</returns>
        public DiscreteChar Complement()
        {
            var result = new DiscreteChar();
            result.probabilityOutsideRanges = this.probabilityOutsideRanges > Eps ? 0 : 1;
            for (int i = 0; i < this.rangeCount; ++i)
            {
                var range = this.ranges[i];
                range.Probability = range.Probability > Eps ? 0 : 1;
                result.AddRange(range);
            }

            result.MergeNeighboringRanges();
            result.Normalize();
            
            return result;
        }

        /// <summary>
        /// Gets a vector of character probabilities under this distribution.
        /// </summary>
        /// <returns>A vector of character probabilities.</returns>
        public PiecewiseVector GetProbs()
        {
            var result = PiecewiseVector.Constant(CharRangeEndExclusive, this.probabilityOutsideRanges);
            for (int i = 0; i < this.rangeCount; ++i)
            {
                var range = this.ranges[i];
                result.Pieces.Add(new ConstantVector(range.StartInclusive, range.EndExclusive - 1, range.Probability));
            }

            return result;
        }

        /// <summary>
        /// Swaps this distribution with a given one.
        /// </summary>
        /// <param name="distribution">The distribution to swap this distribution with.</param>
        public void SwapWith(DiscreteChar distribution)
        {
            Argument.CheckIfNotNull(distribution, "distribution");
            
            Util.Swap(ref this.probabilityOutsideRanges, ref distribution.probabilityOutsideRanges);
            Util.Swap(ref this.ranges, ref distribution.ranges);
            Util.Swap(ref this.rangeCount, ref distribution.rangeCount);
        }

        #endregion

        #region Equals, GetHashCode, ToString

        /// <summary>
        /// Gets a string that represents this distribution.
        /// </summary>
        /// <returns>
        /// A string that represents this distribution.
        /// </returns>
        public override string ToString()
        {
            var resultBuilder = new StringBuilder();
            this.AppendToString(resultBuilder);
            return resultBuilder.ToString();
        }

        /// <summary>
        /// Appends the ToString() for this distribution to the supplied string builder.
        /// </summary>
        /// <param name="stringBuilder">The string builder to append to</param>
        public void AppendToString(StringBuilder stringBuilder)
        {
            if (this.IsPointMass)
            {
                AppendChar(stringBuilder, this.Point);
            }
            else
            {
                for (int i = 0; i < this.rangeCount; ++i)
                {
                    this.ranges[i].AppendToString(stringBuilder);
                    stringBuilder.Append(' ');
                }

                if (stringBuilder.Length > 0)
                {
                    stringBuilder.Append("Otherwise: ");
                }

                stringBuilder.Append(this.probabilityOutsideRanges);
            }
        }

        /// <summary>
        /// Appends a regex expression that represents this character to the supplied string builder.
        /// </summary>
        /// <param name="stringBuilder">The string builder to append to</param>
        /// <param name="useFriendlySymbols">Whether to use friendly symbols."</param>
        public void AppendRegex(StringBuilder stringBuilder, bool useFriendlySymbols = false)
        {
            if (useFriendlySymbols)
            {
                if (this.symbolRepresentation == null)
                {
                    if (this.IsPointMass)
                    {
                        this.symbolRepresentation = this.Point.ToString();
                    }
                    else if (this.IsDigit)
                    {
                        this.symbolRepresentation = DigitSymbolRepresentation;
                    }
                    else if (this.IsLetter)
                    {
                        this.symbolRepresentation = LetterSymbolRepresentation;
                    }
                    else if (this.IsLetterOrDigit)
                    {
                        this.symbolRepresentation = LetterOrDigitSymbolRepresentation;
                    }
                    else if (this.IsLower)
                    {
                        this.symbolRepresentation = LowerSymbolRepresentation;
                    }
                    else if (this.IsUpper)
                    {
                        this.symbolRepresentation = UpperSymbolRepresentation;
                    }
                    else if (this.IsWordChar)
                    {
                        this.symbolRepresentation = WordCharSymbolRepresentation;
                    }
                    else
                    {
                        var representation = new StringBuilder();
                        this.AppendRanges(representation, false);
                        this.symbolRepresentation = representation.ToString();
                    }
                }

                stringBuilder.Append(this.symbolRepresentation);
            }
            else
            {
                if (this.regexRepresentation == null)
                {
                    if (this.IsPointMass)
                    {
                        this.regexRepresentation = Regex.Escape(this.Point.ToString());
                    }
                    else if (this.IsDigit)
                    {
                        this.regexRepresentation = DigitRegexRepresentation;
                    }
                    else if (this.IsLetter)
                    {
                        this.regexRepresentation = LetterRegexRepresentation;
                    }
                    else if (this.IsLetterOrDigit)
                    {
                        this.regexRepresentation = LetterOrDigitRegexRepresentation;
                    }
                    else if (this.IsLower)
                    {
                        this.regexRepresentation = LowerRegexRepresentation;
                    }
                    else if (this.IsUpper)
                    {
                        this.regexRepresentation = UpperRegexRepresentation;
                    }
                    else if (this.IsWordChar)
                    {
                        this.regexRepresentation = WordCharRegexRepresentation;
                    }
                    else
                    {
                        var representation = new StringBuilder();
                        this.AppendRanges(representation, true);
                        this.regexRepresentation = representation.ToString();
                    }
                }

                stringBuilder.Append(this.regexRepresentation);
            }
        }

        /// <summary>
        /// Checks if <paramref name="obj"/> equals to this distribution (i.e. represents the same distribution over characters).
        /// </summary>
        /// <param name="obj">The object to compare this distribution with.</param>
        /// <returns><see langword="true"/> if this distribution is equal to <paramref name="obj"/>, false otherwise.</returns>
        public override bool Equals(object obj)
        {
            var other = obj as DiscreteChar;
            if (other == null)
            {
                return false;
            }
            if (IsPointMass)
            {
                return other.IsPointMass && Point == other.Point;
            }

            return
                (this.charClasses != CharClasses.Unknown && this.charClasses == other.charClasses)
                || this.MaxDiff(other) < Eps;
        }

        /// <summary>
        /// Gets the hash code of this distribution.
        /// </summary>
        /// <returns>The hash code.</returns>
        public override int GetHashCode()
        {
            return 17; // TODO: what to do here? every distribution has multiple representations
        }

        #endregion

        #region Helper methods

        #region Parameter checks

        /// <summary>
        /// Checks if the character ranges in a given list are sorted by range start, non-overlapping,
        /// cover valid characters only and have non-negative (unnormalized) probability values associated with them.
        /// </summary>
        /// <param name="ranges">The list of ranges.</param>
        /// <param name="parameterName">The name of the parameter to report in thrown exceptions.</param>
        private static void CheckRanges(IEnumerable<CharRange> ranges, string parameterName)
        {
            int prevRangeEnd = 0;
            foreach (var range in ranges)
            {
                CheckUnnormalizedProbability(range.Probability, parameterName);
                Argument.CheckIfValid(
                    range.StartInclusive >= prevRangeEnd && range.EndExclusive <= CharRangeEndExclusive,
                    parameterName,
                    "Ranges must only include valid characters and be non-overlapping.");
                prevRangeEnd = range.EndExclusive;
            }
        }

        /// <summary>
        /// Checks if a given (unnormalized) probability value is non-negative.
        /// </summary>
        /// <param name="probability">The probability value to check.</param>
        /// <param name="parameterName">The name of the parameter to report in thrown exceptions.</param>        
        private static void CheckUnnormalizedProbability(double probability, string parameterName)
        {
            Argument.CheckIfInRange(probability >= 0, parameterName, "The (unnormalized) probability must be non-negative.");
        }

        #endregion

        #region Arithmetic helpers

        /// <summary>
        /// Computes the ratio of two numbers, redefining zero divided by zero to be zero.
        /// </summary>
        /// <param name="numerator">The numerator in the ratio.</param>
        /// <param name="denominator">The denominator in the ratio.</param>
        /// <returns>The ratio.</returns>
        private static double DivideProb(double numerator, double denominator)
        {
            if (denominator == 0.0)
            {
                if (numerator != 0.0)
                {
                    throw new DivideByZeroException();
                }

                return 0.0;
            }

            return numerator / denominator;
        }

        /// <summary>
        /// Computes <c>value1 * Log(value2)</c>, redefining zero times negative infinity to be zero.
        /// </summary>
        /// <param name="value1">The first value.</param>
        /// <param name="value2">The second value.</param>
        /// <returns>The computed value.</returns>
        private static double ValueTimesLogValue(double value1, double value2)
        {
            if (value2 == 0)
            {
                if (value1 != 0)
                {
                    return double.NegativeInfinity;
                }

                return 0;
            }

            return value1 * Math.Log(value2);
        }

        #endregion

        #region ToString helpers

        /// <summary>
        /// Appends a string representation of a given character.
        /// </summary>
        /// <param name="stringBuilder">The string builder to append to.</param>
        /// <param name="character">The character.</param>
        private static void AppendChar(StringBuilder stringBuilder, char character)
        {
            if (char.IsControl(character))
            {
                stringBuilder.Append('#');
                stringBuilder.Append((int)character);
            }
            else
            {
                stringBuilder.Append('\'');
                stringBuilder.Append(character);
                stringBuilder.Append('\'');
            }
        }

        private void AppendRanges(StringBuilder representation, bool escape)
        {
            if (this.rangeCount >= 1)
            {
                representation.Append('[');
                for (int i = 0; i < this.rangeCount; ++i)
                {
                    var range = this.ranges[i];
                    AppendCharacterRange(representation, range, escape);
                }

                representation.Append(']');
            }
        }

        private static void AppendCharacterRange(StringBuilder resultBuilder, CharRange range, bool escape)
        {
            var endInclusive = range.EndExclusive - 1;
            var start = ((char)range.StartInclusive).ToString();
            if (escape)
            {
                start = Regex.Escape(start);
            }

            if (endInclusive > range.StartInclusive)
            {
                var end = ((char)(range.EndExclusive - 1)).ToString();
                if (escape)
                {
                    end = Regex.Escape(end);
                }

                resultBuilder.Append(start);
                resultBuilder.Append(@"-");
                resultBuilder.Append(end);
            }
            else
            {
                resultBuilder.Append(start);
            }
        }
        #endregion

        /// <summary>
        /// Adds a new character range to <see cref="ranges"/>, increasing its size if necessary.
        /// </summary>
        /// <param name="range">The range to add.</param>
        private void AddRange(CharRange range)
        {
            if (this.rangeCount == this.ranges.Length)
            {
                var newRanges = new CharRange[this.rangeCount * 2];
                Array.Copy(this.ranges, newRanges, this.rangeCount);
                this.ranges = newRanges;
            }

            this.ranges[this.rangeCount++] = range;
        }

        /// <summary>
        /// Gets the index of the character range containing a given character, if any.
        /// </summary>
        /// <param name="value">The character.</param>
        /// <returns>
        /// The index of the character range containing <paramref name="value"/>,
        /// or <see cref="UnknownRange"/> if no range has been found.
        /// </returns>
        private int GetRangeIndexForCharacter(char value)
        {
            for (int i = 0; i < this.rangeCount; ++i)
            {
                if (this.ranges[i].StartInclusive <= value && this.ranges[i].EndExclusive > value)
                {
                    return i;
                }
            }

            return UnknownRange;
        }

        /// <summary>
        /// Merges neighboring character ranges that have the same associated probability value.
        /// </summary>
        private void MergeNeighboringRanges()
        {
            if (this.rangeCount == 0)
            {
                return;
            }

            var newRanges = new CharRange[this.ranges.Length];
            int newRangeCount = 0;
            for (int i = 0; i < this.rangeCount; ++i)
            {
                var range = this.ranges[i];
                if (Math.Abs(range.Probability - this.probabilityOutsideRanges) < Eps)
                {
                    continue;
                }

                if (newRangeCount > 0)
                {
                    var prevRange = newRanges[newRangeCount - 1];
                    if (range.StartInclusive == prevRange.EndExclusive && Math.Abs(range.Probability - prevRange.Probability) < Eps)
                    {
                        prevRange.EndExclusive = range.EndExclusive;
                        newRanges[newRangeCount - 1] = prevRange;
                        continue;
                    }
                }
                
                newRanges[newRangeCount++] = range;
            }

            this.ranges = newRanges;
            this.rangeCount = newRangeCount;
        }

        /// <summary>
        /// Computes the normalizer of this distribution.
        /// </summary>
        /// <returns>The computed normalizer.</returns>
        private double ComputeNormalizer()
        {
            double normalizer = 0;
            int prevRangeEnd = 0;

            for (int i = 0; i < this.rangeCount; ++i)
            {
                var range = this.ranges[i];
                normalizer += (range.StartInclusive - prevRangeEnd) * this.probabilityOutsideRanges;
                normalizer += (range.EndExclusive - range.StartInclusive) * range.Probability;
                prevRangeEnd = range.EndExclusive;
            }

            normalizer += (CharRangeEndExclusive - prevRangeEnd) * this.probabilityOutsideRanges;

            return normalizer;
        }

        /// <summary>
        /// Normalizes the character probabilities.
        /// </summary>
        private void Normalize()
        {
            double normalizer = this.ComputeNormalizer();
            if (Math.Abs(normalizer) < Eps)
            {
                throw new AllZeroException("A character distribution that is zero everywhere has been produced.");
            }
            
            this.probabilityOutsideRanges /= normalizer;
            for (int i = 0; i < this.rangeCount; ++i)
            {
                this.ranges[i].Probability /= normalizer;
            }
        }

        public static DiscreteChar ToLower(DiscreteChar unnormalizedCharDist)
        {
            switch (unnormalizedCharDist.charClasses)
            {
                case CharClasses.Unknown:
                    var ranges = unnormalizedCharDist.GetRanges();
                    var probVector = PiecewiseVector.Zero(CharRangeEndExclusive);
                    foreach (var range in ranges)
                    {
                        var rangeWeight = range.Probability;
                        for (var ch = range.StartInclusive; ch < range.EndExclusive; ch++)
                        {
                            var transformedChar = char.ToLowerInvariant((char) ch);
                            probVector[transformedChar] += rangeWeight;
                        }
                    }
                    return FromVector(probVector);
                case CharClasses.Digit:
                case CharClasses.Lower:
                    return unnormalizedCharDist;
                case CharClasses.Upper:
                case CharClasses.Letter:
                    return Lower();
                case CharClasses.LetterOrDigit:
                    var result = Lower();
                    result.SetToSum(0.5, Lower(), 0.5, Digit());
                    result.SetToPartialUniform();
                    return result;
                case CharClasses.WordChar:
                case CharClasses.Uniform:
                    throw new NotSupportedException();
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        #endregion

        #region Nested classes

        [Flags]
        public enum CharClasses
        {
            Unknown = 0x0,
            Digit = 1 << 1,
            Lower = 1 << 2,
            Upper = 1 << 3,
            Letter = 1 << 4,
            LetterOrDigit = 1 << 5,
            WordChar = 1 << 6,
            Uniform = 1 << 7,
        }

        public static CharClasses GetImpliedCharClasses(CharClasses charClass)
        {
            switch (charClass)
            {
                case CharClasses.Unknown:
                    return CharClasses.Unknown;
                case CharClasses.Digit:
                    return CharClasses.Digit | GetImpliedCharClasses(CharClasses.LetterOrDigit);
                case CharClasses.Lower:
                    return CharClasses.Lower | GetImpliedCharClasses(CharClasses.Letter);
                case CharClasses.Upper:
                    return CharClasses.Upper | GetImpliedCharClasses(CharClasses.Letter);
                case CharClasses.Letter:
                    return CharClasses.Letter | GetImpliedCharClasses(CharClasses.LetterOrDigit);
                case CharClasses.LetterOrDigit:
                    return CharClasses.LetterOrDigit | GetImpliedCharClasses(CharClasses.WordChar); 
                case CharClasses.WordChar:
                    return CharClasses.WordChar | GetImpliedCharClasses(CharClasses.Uniform);
                case CharClasses.Uniform:
                    return CharClasses.Uniform;
                default:
                    throw new ArgumentOutOfRangeException(nameof(charClass), charClass, null);
            }
        }

        /// <summary>
        /// Represents a range of characters, with an associated probability.
        /// </summary>
        [Serializable]
        [DataContract]
        public struct CharRange
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="CharRange"/> struct
            /// with given range bounds and character probability.
            /// </summary>
            /// <param name="startInclusive">The start of the character range (inclusive).</param>
            /// <param name="endExclusive">The end of the character range (exclusive).</param>
            /// <param name="probability">The probability of characters in the range.</param>
            [Construction("StartInclusive", "EndExclusive", "Probability")]
            public CharRange(int startInclusive, int endExclusive, double probability)
                : this()
            {
                if (startInclusive >= endExclusive && !(endExclusive == 0 && startInclusive == 0))
                    throw new ArgumentException("Inverted character range");
                this.StartInclusive = startInclusive;
                this.EndExclusive = endExclusive;
                this.Probability = probability;
            }

            /// <summary>
            /// Gets or sets the start of the range (inclusive).
            /// </summary>
            [DataMember]
            public int StartInclusive { get; set; }

            /// <summary>
            /// Gets or sets the end of the range (exclusive).
            /// </summary>
            [DataMember]
            public int EndExclusive { get; set; }

            /// <summary>
            /// Gets or sets the probability associated with the range.
            /// </summary>
            [DataMember]
            public double Probability { get; set; }

            /// <summary>
            /// Gets a string that represents this character range.
            /// </summary>
            /// <returns>
            /// A string that represents this character range.
            /// </returns>
            public override string ToString()
            {
                var sb = new StringBuilder();
                this.AppendToString(sb);
                return sb.ToString();                
            }

            internal void AppendToString(StringBuilder stringBuilder)
            {
                stringBuilder.Append('[');
                AppendChar(stringBuilder, (char)this.StartInclusive);

                if (this.StartInclusive == CharRangeEndExclusive)
                {
                    stringBuilder.Append(", end)");
                }
                else
                {
                    stringBuilder.Append(", ");
                    AppendChar(stringBuilder, (char)(this.EndExclusive - 1));
                    stringBuilder.Append(']');
                }

                stringBuilder.Append(": ");
                stringBuilder.Append(this.Probability.ToString("G4"));
            }

            /// <summary>
            /// Writes a char range.
            /// </summary>
            public void Write(Action<int> writeInt32, Action<double> writeDouble)
            {
                writeInt32(StartInclusive);
                writeInt32(EndExclusive);
                writeDouble(Probability);
            }

            /// <summary>
            /// Reads a char range.
            /// </summary>
            public static CharRange Read(Func<int> readInt32, Func<double> readDouble) => new CharRange(readInt32(), readInt32(), readDouble());
        }

        /// <summary>
        /// Represents a pair of character ranges with the same start and end, but different probability values.
        /// </summary>
        private struct CharRangePair
        {
            /// <summary>
            /// Gets or sets the start of the ranges (inclusive).
            /// </summary>
            public int StartInclusive { get; set; }

            /// <summary>
            /// Gets or sets the end of the ranges (exclusive).
            /// </summary>
            public int EndExclusive { get; set; }

            /// <summary>
            /// Gets or sets the probability value associated with the first range.
            /// </summary>
            public double Probability1 { get; set; }

            /// <summary>
            /// Gets or sets the probability value associated with the second range.
            /// </summary>
            public double Probability2 { get; set; }

            /// <summary>
            /// Gets a string that represents this character range.
            /// </summary>
            /// <returns>
            /// A string that represents this character range.
            /// </returns>
            public override string ToString()
            {
                var sb = new StringBuilder();
                this.AppendToString(sb);
                return sb.ToString();
            }

            internal void AppendToString(StringBuilder stringBuilder)
            {
                stringBuilder.Append('[');
                AppendChar(stringBuilder, (char)this.StartInclusive);

                if (this.StartInclusive == CharRangeEndExclusive)
                {
                    stringBuilder.Append(", end)");
                }
                else
                {
                    stringBuilder.Append(", ");
                    AppendChar(stringBuilder, (char)(this.EndExclusive - 1));
                    stringBuilder.Append(']');
                }

                stringBuilder.Append(": ");
                stringBuilder.Append(this.Probability1.ToString("G4"));
                stringBuilder.Append(", ");
                stringBuilder.Append(this.Probability2.ToString("G4"));
            }

            /// <summary>
            /// Gets the intersecting ranges of two distributions.
            /// </summary>
            /// <param name="distribution1">The first distribution.</param>
            /// <param name="distribution2">The second distribution</param>
            /// <param name="excludeZeroProb">Whether to exclude non-intersectng ranges in the case where both distibrutions have zero probability outside their ranges.</param>
            /// <returns></returns>
            public static IEnumerable<CharRangePair> CombinedRanges(DiscreteChar distribution1, DiscreteChar distribution2, bool excludeZeroProb = true)
            {
                if (excludeZeroProb && distribution1.probabilityOutsideRanges == 0.0 && distribution2.probabilityOutsideRanges == 0.0)
                {
                    if (distribution1.rangeCount == 0 || distribution2.rangeCount == 0)
                    {
                        yield break;
                    }

                    int rangeIndex1 = 0;
                    int rangeIndex2 = 0;

                    while (rangeIndex1 < distribution1.rangeCount && rangeIndex2 < distribution2.rangeCount)
                    {
                        var range1 = distribution1.ranges[rangeIndex1];
                        var range2 = distribution2.ranges[rangeIndex2];
                        if (range1.StartInclusive <= range2.StartInclusive)
                        {
                            if (range2.StartInclusive < range1.EndExclusive)
                            {
                                // We have an intersection
                                int endExclusive = Math.Min(range1.EndExclusive, range2.EndExclusive);
                                yield return new CharRangePair()
                                {
                                    StartInclusive = range2.StartInclusive,
                                    EndExclusive = endExclusive,
                                    Probability1 = range1.Probability,
                                    Probability2 = range2.Probability
                                };

                                if (range1.EndExclusive == endExclusive)
                                {
                                    rangeIndex1++;
                                }

                                if (range2.EndExclusive == endExclusive)
                                {
                                    rangeIndex2++;
                                }
                            }
                            else
                            {
                                // Can be no more intersections with range 1
                                rangeIndex1++;
                            }
                        }
                        else
                        {
                            if (range1.StartInclusive < range2.EndExclusive)
                            {
                                int endExclusive = Math.Min(range1.EndExclusive, range2.EndExclusive);
                                yield return new CharRangePair()
                                {
                                    StartInclusive = range1.StartInclusive,
                                    EndExclusive = endExclusive,
                                    Probability1 = range1.Probability,
                                    Probability2 = range2.Probability
                                };

                                if (range1.EndExclusive == endExclusive)
                                {
                                    rangeIndex1++;
                                }

                                if (range2.EndExclusive == endExclusive)
                                {
                                    rangeIndex2++;
                                }
                            }
                            else
                            {
                                // Can be no more intersections with range 2
                                rangeIndex2++;
                            }
                        }
                    }
                }
                else
                {
                    // The implementation of this function has been optimized heavily. If you decide to change something,
                    // please make sure that the performance is unaffected.
                    int currentStartInclusive = 0;
                    int currentEndExclusive = 0;
                    int rangeIndex1 = 0;
                    int rangeIndex2 = 0;
                    double currentProbability1 = 0.0;
                    double currentProbability2 = 0.0;

                    while (currentStartInclusive != CharRangeEndExclusive)
                    {
                        currentStartInclusive = currentEndExclusive;
                        if (currentStartInclusive == CharRangeEndExclusive)
                        {
                            yield break;
                        }

                        if (rangeIndex1 < distribution1.rangeCount &&
                            distribution1.ranges[rangeIndex1].EndExclusive == currentStartInclusive)
                        {
                            ++rangeIndex1;
                        }

                        if (rangeIndex2 < distribution2.rangeCount &&
                            distribution2.ranges[rangeIndex2].EndExclusive == currentStartInclusive)
                        {
                            ++rangeIndex2;
                        }

                        if (rangeIndex1 < distribution1.rangeCount)
                        {
                            if (distribution1.ranges[rangeIndex1].StartInclusive > currentStartInclusive)
                            {
                                currentEndExclusive = distribution1.ranges[rangeIndex1].StartInclusive;
                                currentProbability1 = distribution1.probabilityOutsideRanges;
                            }
                            else
                            {
                                currentEndExclusive = distribution1.ranges[rangeIndex1].EndExclusive;
                                currentProbability1 = distribution1.ranges[rangeIndex1].Probability;
                            }
                        }
                        else
                        {
                            currentProbability1 = distribution1.probabilityOutsideRanges;
                            currentEndExclusive = CharRangeEndExclusive;
                        }

                        if (rangeIndex2 < distribution2.rangeCount)
                        {
                            if (distribution2.ranges[rangeIndex2].StartInclusive > currentStartInclusive)
                            {
                                currentEndExclusive = Math.Min(currentEndExclusive, distribution2.ranges[rangeIndex2].StartInclusive);
                                currentProbability2 = distribution2.probabilityOutsideRanges;
                            }
                            else
                            {
                                currentEndExclusive = Math.Min(currentEndExclusive, distribution2.ranges[rangeIndex2].EndExclusive);
                                currentProbability2 = distribution2.ranges[rangeIndex2].Probability;
                            }
                        }
                        else
                        {
                            currentProbability2 = distribution2.probabilityOutsideRanges;
                        }

                        yield return new CharRangePair()
                        {
                            StartInclusive = currentStartInclusive,
                            EndExclusive = currentEndExclusive,
                            Probability1 = currentProbability1,
                            Probability2 = currentProbability2
                        };
                    }
                }
            }
        }

        /////// <summary>
        /////// Given a pair of character distributions,
        /////// splits the whole character range into non-overlapping sub-ranges where both distributions have constant probability,
        /////// and iterates over these ranges in the order of increasing character codes.
        /////// </summary>
        ////private class CharRangePairIterator
        ////{
        ////    /// <summary>
        ////    /// The first distribution.
        ////    /// </summary>
        ////    private readonly DiscreteChar distribution1;

        ////    /// <summary>
        ////    /// The second distribution.
        ////    /// </summary>
        ////    private readonly DiscreteChar distribution2;

        ////    /// <summary>
        ////    /// The index of the current range in <see cref="distribution1"/>.
        ////    /// </summary>
        ////    private int rangeIndex1;

        ////    /// <summary>
        ////    /// The index of the current range in <see cref="distribution2"/>.
        ////    /// </summary>
        ////    private int rangeIndex2;

        ////    /// <summary>
        ////    /// The character range the iterator currently points to.
        ////    /// </summary>
        ////    private CharRangePair current;

        ////    /// <summary>
        ////    /// Initializes a new instance of the <see cref="CharRangePairIterator"/> class
        ////    /// with a pair of character distributions.
        ////    /// </summary>
        ////    /// <param name="distribution1">The first distribution.</param>
        ////    /// <param name="distribution2">The second distribution.</param>
        ////    public CharRangePairIterator(DiscreteChar distribution1, DiscreteChar distribution2)
        ////    {
        ////        this.distribution1 = distribution1;
        ////        this.distribution2 = distribution2;
        ////    }

        ////    /// <summary>
        ////    /// Gets the character range the iterator currently points to.
        ////    /// </summary>
        ////    public CharRangePair Current
        ////    {
        ////        get { return this.current; }
        ////    }

        ////    /// <summary>
        ////    /// Advances the iterator.
        ////    /// </summary>
        ////    /// <returns>
        ////    /// <see langword="true"/> if the iterator still points to a valid character range,
        ////    /// <see langword="false"/> otherwise.
        ////    /// </returns>
        ////    public bool MoveNext()
        ////    {
        ////        {
        ////            // The implementation of this function has been optimized heavily. If you decide to change something,
        ////            // please make sure that the performance is unaffected.

        ////            this.current.StartInclusive = this.current.EndExclusive;

        ////            if (this.rangeIndex1 < this.distribution1.rangeCount &&
        ////                this.distribution1.ranges[this.rangeIndex1].EndExclusive == this.current.StartInclusive)
        ////            {
        ////                ++this.rangeIndex1;
        ////            }

        ////            if (this.rangeIndex2 < this.distribution2.rangeCount &&
        ////                this.distribution2.ranges[this.rangeIndex2].EndExclusive == this.current.StartInclusive)
        ////            {
        ////                ++this.rangeIndex2;
        ////            }

        ////            if (this.rangeIndex1 < this.distribution1.rangeCount)
        ////            {
        ////                if (this.distribution1.ranges[this.rangeIndex1].StartInclusive > this.current.StartInclusive)
        ////                {
        ////                    this.current.EndExclusive = this.distribution1.ranges[this.rangeIndex1].StartInclusive;
        ////                    this.current.Probability1 = this.distribution1.probabilityOutsideRanges;
        ////                }
        ////                else
        ////                {
        ////                    this.current.EndExclusive = this.distribution1.ranges[this.rangeIndex1].EndExclusive;
        ////                    this.current.Probability1 = this.distribution1.ranges[this.rangeIndex1].Probability;
        ////                }
        ////            }
        ////            else
        ////            {
        ////                this.current.Probability1 = this.distribution1.probabilityOutsideRanges;
        ////                this.current.EndExclusive = CharRangeEndExclusive;
        ////            }

        ////            if (this.rangeIndex2 < this.distribution2.rangeCount)
        ////            {
        ////                if (this.distribution2.ranges[this.rangeIndex2].StartInclusive > this.current.StartInclusive)
        ////                {
        ////                    this.current.EndExclusive = Math.Min(this.current.EndExclusive, this.distribution2.ranges[this.rangeIndex2].StartInclusive);
        ////                    this.current.Probability2 = this.distribution2.probabilityOutsideRanges;
        ////                }
        ////                else
        ////                {
        ////                    this.current.EndExclusive = Math.Min(this.current.EndExclusive, this.distribution2.ranges[this.rangeIndex2].EndExclusive);
        ////                    this.current.Probability2 = this.distribution2.ranges[this.rangeIndex2].Probability;
        ////                }
        ////            }
        ////            else
        ////            {
        ////                this.current.Probability2 = this.distribution2.probabilityOutsideRanges;
        ////            }

        ////            return this.current.StartInclusive != CharRangeEndExclusive;
        ////        }
        ////    }
        ////}

        #endregion

        /// <summary>
        /// Writes a discrete character.
        /// </summary>
        public void Write(Action<int> writeInt32, Action<double> writeDouble)
        {
            var propertyMask = new BitVector32();
            var idx = 0;
            propertyMask[1 << idx++] = ranges != null;
            writeInt32(propertyMask.Data);
            if (ranges != null)
            {
                writeInt32(ranges.Length);
                ranges.ForEach(range => range.Write(writeInt32, writeDouble));
            }
            writeInt32(rangeCount);
            writeInt32((int)charClasses);
            writeDouble(probabilityOutsideRanges);
        }

        /// <summary>
        /// Reads a discrete character.
        /// </summary>
        public static DiscreteChar Read(Func<int> readInt32, Func<double> readDouble)
        {
            var propertyMask = new BitVector32(readInt32());
            var res = new DiscreteChar();
            var idx = 0;
            var hasRanges = propertyMask[1 << idx++];

            if (hasRanges)
            {
                var nRanges = readInt32();
                var ranges = new CharRange[nRanges];
                for (var i = 0; i < nRanges; i++)
                {
                    ranges[i] = CharRange.Read(readInt32, readDouble);
                }
                res.ranges = ranges;
            }
            res.rangeCount = readInt32();
            res.charClasses = (CharClasses)readInt32();
            res.probabilityOutsideRanges = readDouble();
            return res;
        }
    }
}