// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections;

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using System.Runtime.Serialization;
    using System.Text;
    using System.Text.RegularExpressions;

    using Microsoft.ML.Probabilistic.Distributions.Automata;
    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Utilities;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// Represents a distribution over characters.
    /// </summary>
    [Quality(QualityBand.Experimental)]
    [Serializable]
    public struct DiscreteChar :
        IDistribution<char>,
        SettableTo<DiscreteChar>, SettableTo<ImmutableDiscreteChar>,
        SettableToProduct<DiscreteChar>, SettableToProduct<ImmutableDiscreteChar>,
        SettableToProduct<ImmutableDiscreteChar, DiscreteChar>, SettableToProduct<DiscreteChar, ImmutableDiscreteChar>,
        SettableToRatio<DiscreteChar>, SettableToRatio<ImmutableDiscreteChar>,
        SettableToRatio<ImmutableDiscreteChar, DiscreteChar>, SettableToRatio<DiscreteChar, ImmutableDiscreteChar>,
        SettableToPower<DiscreteChar>, SettableToPower<ImmutableDiscreteChar>,
        SettableToWeightedSumExact<DiscreteChar>, SettableToWeightedSumExact<ImmutableDiscreteChar>,
        SettableToPartialUniform<DiscreteChar>, SettableToPartialUniform<ImmutableDiscreteChar>,
        CanGetLogAverageOf<DiscreteChar>, CanGetLogAverageOf<ImmutableDiscreteChar>,
        CanGetLogAverageOfPower<DiscreteChar>, CanGetLogAverageOfPower<ImmutableDiscreteChar>,
        CanGetAverageLog<DiscreteChar>, CanGetAverageLog<ImmutableDiscreteChar>,
        CanGetMode<char>,
        Sampleable<char>,
        CanEnumerateSupport<char>,
        IEquatable<DiscreteChar>
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
        /// The (exclusive) end of the character range.
        /// </summary>
        public const int CharRangeEndExclusive = char.MaxValue + 1;

        #endregion

        #region Data

        /// <summary>
        /// Immutable snapshot of the current distribution.
        /// </summary>
        [DataMember]
        public ImmutableDiscreteChar WrappedDistribution { get; private set; }

        #endregion

        #region Constructors

        [Construction("WrappedDistribution")]
        public DiscreteChar(ImmutableDiscreteChar wrappedDistribution)
        {
            WrappedDistribution = wrappedDistribution;
        }

        #endregion

        #region Properties

        public bool IsInitialized => WrappedDistribution.IsInitialized;

        #region Properties matching factory methods

        /// <summary>
        /// Gets a value indicating whether this distribution equals the distribution created by <see cref="Digit"/>.
        /// </summary>
        public bool IsDigit => WrappedDistribution.IsDigit;

        /// <summary>
        /// Gets a value indicating whether this distribution equals the distribution created by <see cref="Lower"/>.
        /// </summary>
        public bool IsLower => WrappedDistribution.IsLower;

        /// <summary>
        /// Gets a value indicating whether this distribution equals the distribution created by <see cref="Upper"/>.
        /// </summary>
        public bool IsUpper => WrappedDistribution.IsUpper;

        /// <summary>
        /// Gets a value indicating whether this distribution equals the distribution created by <see cref="Letter"/>.
        /// </summary>
        public bool IsLetter => WrappedDistribution.IsLetter;

        /// <summary>
        /// Gets a value indicating whether this distribution equals the distribution created by <see cref="LetterOrDigit"/>.
        /// </summary>
        public bool IsLetterOrDigit => WrappedDistribution.IsLetterOrDigit;

        /// <summary>
        /// Gets a value indicating whether this distribution equals the distribution created by <see cref="WordChar"/>.
        /// </summary>
        public bool IsWordChar => WrappedDistribution.IsWordChar;

        #endregion

        /// <summary>
        /// Gets the probability of a given character under this distribution.
        /// </summary>
        /// <param name="value">The character.</param>
        /// <returns>The probability of the character under this distribution.</returns>
        public double this[char value] => WrappedDistribution[value];

        #endregion

        #region Factory methods

        /// <inheritdoc cref="ImmutableDiscreteChar.Create(IEnumerable{ImmutableDiscreteChar.CharRange})"/>
        public static DiscreteChar Create(IEnumerable<ImmutableDiscreteChar.CharRange> ranges) =>
            new DiscreteChar(ImmutableDiscreteChar.Create(ranges));

        /// <inheritdoc cref="ImmutableDiscreteChar.Create(ReadOnlyArray{ImmutableDiscreteChar.CharRange})"/>
        public static DiscreteChar Create(ReadOnlyArray<ImmutableDiscreteChar.CharRange> ranges) =>
            new DiscreteChar(ImmutableDiscreteChar.Create(ranges));

        /// <inheritdoc cref="ImmutableDiscreteChar.Uniform"/>
        [Construction(UseWhen = "IsUniform")]
        public static DiscreteChar Uniform() => new DiscreteChar(ImmutableDiscreteChar.Uniform());

        /// <inheritdoc cref="ImmutableDiscreteChar.Digit"/>
        [Construction(UseWhen = "IsDigit")]
        public static DiscreteChar Digit() => new DiscreteChar(ImmutableDiscreteChar.Digit());

        /// <inheritdoc cref="ImmutableDiscreteChar.Lower"/>
        [Construction(UseWhen = "IsLower")]
        public static DiscreteChar Lower() => new DiscreteChar(ImmutableDiscreteChar.Lower());

        /// <inheritdoc cref="ImmutableDiscreteChar.Upper"/>
        [Construction(UseWhen = "IsUpper")]
        public static DiscreteChar Upper() => new DiscreteChar(ImmutableDiscreteChar.Upper());

        /// <inheritdoc cref="ImmutableDiscreteChar.Letter"/>
        [Construction(UseWhen = "IsLetter")]
        public static DiscreteChar Letter() => new DiscreteChar(ImmutableDiscreteChar.Letter());

        /// <inheritdoc cref="ImmutableDiscreteChar.LetterOrDigit"/>
        [Construction(UseWhen = "IsLetterOrDigit")]
        public static DiscreteChar LetterOrDigit() => new DiscreteChar(ImmutableDiscreteChar.LetterOrDigit());

        /// <inheritdoc cref="ImmutableDiscreteChar.WordChar"/>
        [Construction(UseWhen = "IsWordChar")]
        public static DiscreteChar WordChar() => new DiscreteChar(ImmutableDiscreteChar.WordChar());

        /// <inheritdoc cref="ImmutableDiscreteChar.NonWordChar"/>
        public static DiscreteChar NonWordChar() => new DiscreteChar(ImmutableDiscreteChar.NonWordChar());

        /// <inheritdoc cref="ImmutableDiscreteChar.Whitespace"/>
        public static DiscreteChar Whitespace() => new DiscreteChar(ImmutableDiscreteChar.Whitespace());

        /// <inheritdoc cref="ImmutableDiscreteChar.Any"/>
        public static DiscreteChar Any() => new DiscreteChar(ImmutableDiscreteChar.Any());

        /// <inheritdoc cref="ImmutableDiscreteChar.InRange(char, char)"/>
        public static DiscreteChar InRange(char start, char end) => new DiscreteChar(ImmutableDiscreteChar.InRange(start, end));

        /// <inheritdoc cref="ImmutableDiscreteChar.InRanges(char[])"/>
        public static DiscreteChar InRanges(params char[] startEndPairs) => new DiscreteChar(ImmutableDiscreteChar.InRanges(startEndPairs));

        /// <inheritdoc cref="ImmutableDiscreteChar.InRanges(IEnumerable{char})"/>
        public static DiscreteChar InRanges(IEnumerable<char> startEndPairs) => new DiscreteChar(ImmutableDiscreteChar.InRanges(startEndPairs));

        /// <inheritdoc cref="ImmutableDiscreteChar.OneOf(char[])"/>
        public static DiscreteChar OneOf(params char[] chars) => new DiscreteChar(ImmutableDiscreteChar.OneOf(chars));

        /// <inheritdoc cref="ImmutableDiscreteChar.OneOf(IEnumerable{char})"/>
        public static DiscreteChar OneOf(IEnumerable<char> chars) => new DiscreteChar(ImmutableDiscreteChar.OneOf(chars));

        /// <inheritdoc cref="ImmutableDiscreteChar.UniformOver(char[])"/>
        public static DiscreteChar UniformOver(params char[] chars) => new DiscreteChar(ImmutableDiscreteChar.UniformOver(chars));

        /// <inheritdoc cref="ImmutableDiscreteChar.UniformOver(IEnumerable{char})"/>
        public static DiscreteChar UniformOver(IEnumerable<char> chars) => new DiscreteChar(ImmutableDiscreteChar.UniformOver(chars));

        /// <inheritdoc cref="ImmutableDiscreteChar.UniformInRange(char, char)"/>
        public static DiscreteChar UniformInRange(char start, char end) => new DiscreteChar(ImmutableDiscreteChar.UniformInRanges(start, end));

        /// <inheritdoc cref="ImmutableDiscreteChar.UniformInRanges(char[])"/>
        public static DiscreteChar UniformInRanges(params char[] startEndPairs) => new DiscreteChar(ImmutableDiscreteChar.UniformInRanges(startEndPairs));

        /// <inheritdoc cref="ImmutableDiscreteChar.UniformInRanges(IEnumerable{char})"/>
        public static DiscreteChar UniformInRanges(IEnumerable<char> startEndPairs) => new DiscreteChar(ImmutableDiscreteChar.UniformInRanges(startEndPairs));

        /// <inheritdoc cref="ImmutableDiscreteChar.PointMass(char)"/>
        public static DiscreteChar PointMass(char point) => new DiscreteChar(ImmutableDiscreteChar.PointMass(point));

        /// <inheritdoc cref="ImmutableDiscreteChar.FromVector(Vector)"/>
        public static DiscreteChar FromVector(Vector vector) => new DiscreteChar(ImmutableDiscreteChar.FromVector(vector));

        /// <inheritdoc cref="ImmutableDiscreteChar.FromVector(PiecewiseVector)"/>
        public static DiscreteChar FromVector(PiecewiseVector vector) => new DiscreteChar(ImmutableDiscreteChar.FromVector(vector));

        #endregion

        #region Equality, ToString, GetHashCode

        /// <summary>
        /// Checks if <paramref name="that"/> equals to this distribution (i.e. represents the same distribution over characters).
        /// </summary>
        /// <param name="that">The object to compare this distribution with.</param>
        /// <returns><see langword="true"/> if this distribution is equal to <paramref name="that"/>, false otherwise.</returns>
        public bool Equals(DiscreteChar that) => WrappedDistribution.Equals(that.WrappedDistribution);

        public override bool Equals(object obj) => obj is DiscreteChar that && WrappedDistribution.Equals(that.WrappedDistribution);

        public override int GetHashCode() => WrappedDistribution.GetHashCode();

        public override string ToString() => WrappedDistribution.ToString();

        public void AppendToString(StringBuilder stringBuilder) => WrappedDistribution.AppendToString(stringBuilder);

        /// <inheritdoc cref="ImmutableDiscreteChar.AppendRegex(StringBuilder, bool)"/>
        public void AppendRegex(StringBuilder stringBuilder, bool useFriendlySymbols = false) =>
            WrappedDistribution.AppendRegex(stringBuilder, useFriendlySymbols);

        #endregion

        #region Distribution interface implementation

        #region IDistribution implementation

        /// <inheritdoc cref="Clone"/>
        object ICloneable.Clone() => Clone();

        /// <summary>
        /// Creates a copy of this distribution.
        /// </summary>
        /// <returns>The created copy.</returns>
        public DiscreteChar Clone() => new DiscreteChar(WrappedDistribution);

        /// <inheritdoc/>
        public bool IsUniform() => WrappedDistribution.IsUniform();

        /// <inheritdoc cref="ImmutableDiscreteChar.MaxDiff(object)"/>
        public double MaxDiff(object that)
        {
            Argument.CheckIfNotNull(that, nameof(that));
            return that is DiscreteChar thatDist
                ? WrappedDistribution.MaxDiff(thatDist.WrappedDistribution)
                : double.PositiveInfinity;
        }

        /// <inheritdoc/>
        public void SetToUniform()
        {
            WrappedDistribution = WrappedDistribution.CreateUniform();
        }

        #endregion

        #region IDistribution<T> implementation

        /// <summary>
        /// Gets or sets the point mass represented by the distribution.
        /// </summary>
        public char Point
        {
            get => WrappedDistribution.Point;
            set => WrappedDistribution = WrappedDistribution.CreatePointMass(value);
        }

        /// <inheritdoc/>
        public bool IsPointMass => WrappedDistribution.IsPointMass;

        /// <inheritdoc/>
        public double GetLogProb(char value) => WrappedDistribution.GetLogProb(value);

        #endregion

        #region SettableTo implementation

        /// <inheritdoc/>
        public void SetTo(ImmutableDiscreteChar value)
        {
            WrappedDistribution = value;
        }

        /// <inheritdoc/>
        public void SetTo(DiscreteChar value)
        {
            WrappedDistribution = value.WrappedDistribution;
        }

        #endregion

        #region SettableToPartialUniform implementation

        /// <inheritdoc/>
        public void SetToPartialUniform()
        {
            WrappedDistribution = WrappedDistribution.CreatePartialUniform();
        }

        /// <inheritdoc/>
        public void SetToPartialUniformOf(ImmutableDiscreteChar dist)
        {
            WrappedDistribution = dist.CreatePartialUniform();
        }

        /// <inheritdoc/>
        public bool IsPartialUniform() => WrappedDistribution.IsPartialUniform();

        /// <inheritdoc/>
        public void SetToPartialUniformOf(DiscreteChar dist)
        {
            WrappedDistribution = dist.WrappedDistribution.CreatePartialUniform();
        }

        #endregion

        #region CanGetMode implementation

        /// <inheritdoc cref="ImmutableDiscreteChar.GetMode"/>
        public char GetMode() => WrappedDistribution.GetMode();

        #endregion

        #region CanGetLogAverageOf implementation

        /// <inheritdoc/>
        public double GetLogAverageOf(ImmutableDiscreteChar that) => WrappedDistribution.GetLogAverageOf(that);

        /// <inheritdoc/>
        public double GetLogAverageOf(DiscreteChar that) =>
            WrappedDistribution.GetLogAverageOf(that.WrappedDistribution);

        #endregion

        #region CanGetLogAverageOfPower implementation

        /// <inheritdoc/>
        public double GetLogAverageOfPower(ImmutableDiscreteChar that, double power) => WrappedDistribution.GetLogAverageOfPower(that, power);

        /// <inheritdoc/>
        public double GetLogAverageOfPower(DiscreteChar that, double power) =>
            WrappedDistribution.GetLogAverageOfPower(that.WrappedDistribution, power);

        #endregion

        #region CanGetAverageLog implementation

        /// <inheritdoc/>
        public double GetAverageLog(ImmutableDiscreteChar that) => WrappedDistribution.GetAverageLog(that);

        /// <inheritdoc/>
        public double GetAverageLog(DiscreteChar that) =>
            WrappedDistribution.GetAverageLog(that.WrappedDistribution);

        #endregion

        #region CanEnumerateSupport implementation

        /// <inheritdoc/>
        public IEnumerable<char> EnumerateSupport() => WrappedDistribution.EnumerateSupport();

        #endregion

        #region SettableToProduct implementation

        /// <inheritdoc/>
        public void SetToProduct(ImmutableDiscreteChar a, ImmutableDiscreteChar b)
        {
            WrappedDistribution = a.Multiply(b);
        }

        /// <inheritdoc/>
        public void SetToProduct(DiscreteChar a, DiscreteChar b)
        {
            WrappedDistribution = a.WrappedDistribution.Multiply(b.WrappedDistribution);
        }

        /// <inheritdoc/>
        public void SetToProduct(ImmutableDiscreteChar a, DiscreteChar b)
        {
            WrappedDistribution = a.Multiply(b.WrappedDistribution);
        }

        /// <inheritdoc/>
        public void SetToProduct(DiscreteChar a, ImmutableDiscreteChar b)
        {
            WrappedDistribution = a.WrappedDistribution.Multiply(b);
        }

        #endregion

        #region SettableToRatio implementation

        /// <inheritdoc/>
        public void SetToRatio(ImmutableDiscreteChar a, ImmutableDiscreteChar b, bool forceProper = false)
        {
            WrappedDistribution = a.Divide(b, forceProper);
        }

        /// <inheritdoc/>
        public void SetToRatio(DiscreteChar a, DiscreteChar b, bool forceProper = false)
        {
            WrappedDistribution = a.WrappedDistribution.Divide(b.WrappedDistribution, forceProper);
        }

        /// <inheritdoc/>
        public void SetToRatio(ImmutableDiscreteChar a, DiscreteChar b, bool forceProper = false)
        {
            WrappedDistribution = a.Divide(b.WrappedDistribution, forceProper);
        }

        /// <inheritdoc/>
        public void SetToRatio(DiscreteChar a, ImmutableDiscreteChar b, bool forceProper = false)
        {
            WrappedDistribution = a.WrappedDistribution.Divide(b, forceProper);
        }

        #endregion

        #region SettableToPower implementation

        /// <inheritdoc/>
        public void SetToPower(ImmutableDiscreteChar value, double exponent)
        {
            WrappedDistribution = value.Pow(exponent);
        }
        
        /// <inheritdoc/>
        public void SetToPower(DiscreteChar value, double exponent)
        {
            WrappedDistribution = value.WrappedDistribution.Pow(exponent);
        }

        #endregion

        #region SettableToWeightedSumExact implementation

        /// <summary>
        /// Replaces the current distribution with a mixture of a given pair of distributions.
        /// </summary>
        /// <param name="weight1">The weight of the first distribution.</param>
        /// <param name="value1">The first distribution.</param>
        /// <param name="weight2">The weight of the second distribution.</param>
        /// <param name="value2">The second distribution.</param>
        public void SetToSum(double weight1, ImmutableDiscreteChar value1, double weight2, ImmutableDiscreteChar value2)
        {
            WrappedDistribution = value1.Sum(weight1, value2, weight2);
        }

        /// <summary>
        /// Replaces the current distribution with a mixture of a given pair of distributions.
        /// </summary>
        /// <param name="weight1">The weight of the first distribution.</param>
        /// <param name="value1">The first distribution.</param>
        /// <param name="weight2">The weight of the second distribution.</param>
        /// <param name="value2">The second distribution.</param>
        public void SetToSum(double weight1, DiscreteChar value1, double weight2, DiscreteChar value2)
        {
            WrappedDistribution = value1.WrappedDistribution.Sum(weight1, value2.WrappedDistribution, weight2);
        }

        #endregion

        #region Sampleable implementation

        /// <inheritdoc/>
        public char Sample() => WrappedDistribution.Sample();

        /// <inheritdoc/>
        public char Sample(char result) => WrappedDistribution.Sample(result);

        #endregion

        #endregion

        #region Character distribution specific interface

        /// <inheritdoc cref="ImmutableDiscreteChar.Ranges"/>
        public ReadOnlyArray<ImmutableDiscreteChar.CharRange> Ranges => WrappedDistribution.Ranges;

        /// <inheritdoc cref="ImmutableDiscreteChar.Complement"/>
        public DiscreteChar Complement() => new DiscreteChar(WrappedDistribution.Complement());

        public static DiscreteChar ToLower(DiscreteChar unnormalizedCharDist) =>
            new DiscreteChar(ImmutableDiscreteChar.ToLower(unnormalizedCharDist.WrappedDistribution));

        /// <inheritdoc cref="ImmutableDiscreteChar.GetProbs"/>
        public PiecewiseVector GetProbs() => WrappedDistribution.GetProbs();

        #endregion

        #region Serialization

        /// <summary>
        /// Writes a discrete character.
        /// </summary>
        public void Write(Action<int> writeInt32, Action<double> writeDouble) =>
            WrappedDistribution.Write(writeInt32, writeDouble);

        /// <summary>
        /// Reads a discrete character.
        /// </summary>
        public static DiscreteChar Read(Func<int> readInt32, Func<double> readDouble) =>
            new DiscreteChar(ImmutableDiscreteChar.Read(readInt32, readDouble));

        #endregion
    }

    /// <summary>
    /// Represents a distribution over characters.
    /// </summary>
    [Quality(QualityBand.Experimental)]
    [Serializable]
    public struct ImmutableDiscreteChar :
        IImmutableDistribution<char, ImmutableDiscreteChar>,
        CanCreatePartialUniform<ImmutableDiscreteChar>,
        CanGetMode<char>,
        CanGetLogAverageOf<ImmutableDiscreteChar>,
        CanGetLogAverageOfPower<ImmutableDiscreteChar>,
        CanGetAverageLog<ImmutableDiscreteChar>,
        CanEnumerateSupport<char>,
        CanComputeProduct<ImmutableDiscreteChar>,
        CanComputeRatio<ImmutableDiscreteChar>,
        CanComputePower<ImmutableDiscreteChar>,
        SummableExactly<ImmutableDiscreteChar>,
        Sampleable<char>,
        ISerializable,
        IEquatable<ImmutableDiscreteChar>
    {
        #region Constants

        /// <summary>
        /// The tolerance value for probability comparisons.
        /// </summary>
        private const double Eps = 1e-15;

        /// <summary>
        /// The probability of a character under a uniform distribution over characters.
        /// </summary>
        private static Weight UniformProb => Weight.Inverse(Weight.FromValue(DiscreteChar.CharRangeEndExclusive));

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

        #region Constructors

        /// <summary>
        /// Stores reference to <see cref="Storage"/> for this distribution.
        /// </summary>
        /// <remarks>
        /// Can be null and shouldn't be used directly. Use <see cref="Data"/> property.
        /// </remarks>
        [DataMember]
        private Storage data_;

        /// <summary>
        /// Gets or sets reference to <see cref="Storage"/> for this distribution.
        /// Getter always returns non-null reference.
        /// </summary>
        private Storage Data
        {
            get => this.data_ ?? StorageCache.Uniform;
            set => this.data_ = value;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="DiscreteChar"/> class
        /// with a given list of constant probability character ranges.
        /// </summary>
        /// <param name="ranges">The constant-probability character ranges.</param>
        /// <param name="rangeCount">The number of valid elements in the <paramref name="ranges"/> array.</param>
        /// <remarks>
        /// The probabilities need to be normalized. The character ranges need to be sorted.
        /// The created objects takes ownership of the character range list.
        /// </remarks>
        private ImmutableDiscreteChar(ReadOnlyArray<CharRange> ranges, int rangeCount) =>
            this.data_ = Storage.Create(ranges);

        private ImmutableDiscreteChar(Storage storage) => this.data_ = storage;

        #endregion

        #region Properties

        public bool IsInitialized => this.data_ != null;

        #region Properties matching factory methods

        /// <summary>
        /// Gets a value indicating whether this distribution equals the distribution created by <see cref="Digit"/>.
        /// </summary>
        public bool IsDigit => this.Data.IsDigit;

        /// <summary>
        /// Gets a value indicating whether this distribution equals the distribution created by <see cref="Lower"/>.
        /// </summary>
        public bool IsLower => this.Data.IsLower;

        /// <summary>
        /// Gets a value indicating whether this distribution equals the distribution created by <see cref="Upper"/>.
        /// </summary>
        public bool IsUpper => this.Data.IsUpper;

        /// <summary>
        /// Gets a value indicating whether this distribution equals the distribution created by <see cref="Letter"/>.
        /// </summary>
        public bool IsLetter => this.Data.IsLetter;

        /// <summary>
        /// Gets a value indicating whether this distribution equals the distribution created by <see cref="LetterOrDigit"/>.
        /// </summary>
        public bool IsLetterOrDigit => this.Data.IsLetterOrDigit;

        /// <summary>
        /// Gets a value indicating whether this distribution equals the distribution created by <see cref="WordChar"/>.
        /// </summary>
        public bool IsWordChar => this.Data.IsWordChar;

        #endregion

        #region Distribution properties

        /// <summary>
        /// Gets the point mass represented by the distribution.
        /// </summary>
        public char Point => this.Data.Point ?? throw new InvalidOperationException();

        /// <inheritdoc/>
        public bool IsPointMass => this.Data.Point.HasValue;

        /// <summary>
        /// Gets the probability of a given character under this distribution.
        /// </summary>
        /// <param name="value">The character.</param>
        /// <returns>The probability of the character under this distribution.</returns>
        public double this[char value] => this.FindProb(value).Value;

        #endregion

        #endregion

        #region Factory methods

        /// <summary>
        /// Creates a distribution given a list of constant probability character ranges.
        /// </summary>
        /// <param name="ranges">The constant-probability character ranges.</param>
        /// <remarks>The probabilities do not need to be normalized. The character ranges do not need to be sorted.</remarks>
        /// <returns>The created distribution.</returns>
        [Construction("Ranges")]
        public static ImmutableDiscreteChar Create(ReadOnlyArray<CharRange> ranges)
        {
            Argument.CheckIfNotNull(ranges, "ranges");

            var builder = StorageBuilder.Create();
            foreach (var range in ranges)
            {
                builder.AddRange(range);
            }
            builder.SortAndCheckRanges();
            return new ImmutableDiscreteChar(builder.GetResult());
        }

        /// <summary>
        /// Creates a distribution given a list of constant probability character ranges.
        /// </summary>
        /// <param name="ranges">The constant-probability character ranges.</param>
        /// <remarks>The probabilities do not need to be normalized. The character ranges do not need to be sorted.</remarks>
        /// <returns>The created distribution.</returns>
        public static ImmutableDiscreteChar Create(IEnumerable<CharRange> ranges)
        {
            Argument.CheckIfNotNull(ranges, "ranges");

            var builder = StorageBuilder.Create();
            foreach (var range in ranges)
            {
                builder.AddRange(range);
            }
            builder.SortAndCheckRanges();
            return new ImmutableDiscreteChar(builder.GetResult());
        }

        /// <summary>
        /// Creates a uniform distribution over characters.
        /// </summary>
        /// <returns>The created distribution.</returns>
        [Construction(UseWhen = "IsUniform")]
        public static ImmutableDiscreteChar Uniform() => new ImmutableDiscreteChar(StorageCache.Uniform);

        /// <summary>
        /// Creates a uniform distribution over digits '0'..'9'.
        /// </summary>
        /// <returns>The created distribution.</returns>
        [Construction(UseWhen = "IsDigit")]
        public static ImmutableDiscreteChar Digit() => new ImmutableDiscreteChar(StorageCache.Digit);

        /// <summary>
        /// Creates a uniform distribution over lowercase letters.
        /// </summary>
        /// <returns>The created distribution.</returns>
        [Construction(UseWhen = "IsLower")]
        public static ImmutableDiscreteChar Lower() => new ImmutableDiscreteChar(StorageCache.Lower);

        /// <summary>
        /// Creates a uniform distribution over uppercase letters.
        /// </summary>
        /// <returns>The created distribution.</returns>
        [Construction(UseWhen = "IsUpper")]
        public static ImmutableDiscreteChar Upper() => new ImmutableDiscreteChar(StorageCache.Upper);

        /// <summary>
        /// Creates a uniform distribution over letters.
        /// </summary>
        /// <returns>The created distribution.</returns>
        [Construction(UseWhen = "IsLetter")]
        public static ImmutableDiscreteChar Letter() => new ImmutableDiscreteChar(StorageCache.Letter);

        /// <summary>
        /// Creates a uniform distribution over letters and '0'..'9'.
        /// </summary>
        /// <returns>The created distribution.</returns>
        [Construction(UseWhen = "IsLetterOrDigit")]
        public static ImmutableDiscreteChar LetterOrDigit() => new ImmutableDiscreteChar(StorageCache.LetterOrDigit);

        /// <summary>
        /// Creates a uniform distribution over word characters (letter, digit and '_').
        /// </summary>
        /// <returns>The created distribution.</returns>
        [Construction(UseWhen = "IsWordChar")]
        public static ImmutableDiscreteChar WordChar() => new ImmutableDiscreteChar(StorageCache.WordChar);

        /// <summary>
        /// Creates a uniform distribution over all characters except (letter, digit and '_').
        /// </summary>
        /// <returns>The created distribution.</returns>
        public static ImmutableDiscreteChar NonWordChar() => new ImmutableDiscreteChar(StorageCache.NonWordChar);

        /// <summary>
        /// Creates a uniform distribution over whitespace characters ('\t'..'\r', ' ').
        /// </summary>
        /// <returns>The created distribution.</returns>
        public static ImmutableDiscreteChar Whitespace() => new ImmutableDiscreteChar(StorageCache.Whitespace);

        /// <summary>
        /// Creates a uniform distribution over all characters.
        /// </summary>
        /// <returns>The created distribution.</returns>
        public static ImmutableDiscreteChar Any() => ImmutableDiscreteChar.Uniform();

        /// <summary>
        /// Creates a uniform distribution over characters in a given range.
        /// </summary>
        /// <param name="start">The start of the range (inclusive).</param>
        /// <param name="end">The end of the range (inclusive).</param>
        /// <returns>The created distribution.</returns>
        public static ImmutableDiscreteChar InRange(char start, char end) => ImmutableDiscreteChar.UniformInRanges(start, end);

        /// <summary>
        /// Creates a distribution which is uniform over values in
        /// multiple ranges specified by pairs of start and end values. These pairs
        /// are specified as adjacent values in an array whose length must therefore be even.
        /// </summary>
        /// <param name="startEndPairs">The array of range starts and ends.</param>
        /// <returns>The created distribution.</returns>
        public static ImmutableDiscreteChar InRanges(params char[] startEndPairs) => ImmutableDiscreteChar.UniformInRanges(startEndPairs);

        /// <summary>
        /// Creates a distribution which is uniform over values in
        /// multiple ranges specified by pairs of start and end values. These pairs
        /// are specified as adjacent values in a sequence whose length must therefore be even.
        /// </summary>
        /// <param name="startEndPairs">The sequence of range starts and ends.</param>
        /// <returns>The created distribution.</returns>
        public static ImmutableDiscreteChar InRanges(IEnumerable<char> startEndPairs) => ImmutableDiscreteChar.UniformInRanges(startEndPairs);

        /// <summary>
        /// Creates a distribution which is uniform over the specified set of characters.
        /// </summary>
        /// <param name="chars">The characters.</param>
        /// <returns>The created distribution.</returns>
        public static ImmutableDiscreteChar OneOf(params char[] chars) => ImmutableDiscreteChar.UniformOver(chars);

        /// <summary>
        /// Creates a distribution which is uniform over the specified set of characters.
        /// </summary>
        /// <param name="chars">The characters.</param>
        /// <returns>The created distribution.</returns>
        public static ImmutableDiscreteChar OneOf(IEnumerable<char> chars) => ImmutableDiscreteChar.UniformOver(chars);

        /// <summary>
        /// Creates a distribution which is uniform over the specified set of characters.
        /// </summary>
        /// <param name="chars">The characters.</param>
        /// <returns>The created distribution.</returns>
        public static ImmutableDiscreteChar UniformOver(params char[] chars) => UniformOver((IEnumerable<char>)chars);

        /// <summary>
        /// Creates a distribution which is uniform over the specified set of characters.
        /// </summary>
        /// <param name="chars">The characters.</param>
        /// <returns>The created distribution.</returns>
        public static ImmutableDiscreteChar UniformOver(IEnumerable<char> chars)
        {
            Argument.CheckIfNotNull(chars, nameof(chars));
            return Create(chars.Select(c => new CharRange(c, c + 1, Weight.One)));
        }

        /// <summary>
        /// Creates a uniform distribution over characters in a given range.
        /// </summary>
        /// <param name="start">The start of the range (inclusive).</param>
        /// <param name="end">The end of the range (inclusive).</param>
        /// <returns>The created distribution.</returns>
        public static ImmutableDiscreteChar UniformInRange(char start, char end) => UniformInRanges(start, end);

        /// <summary>
        /// Creates a distribution which is uniform over values in
        /// multiple ranges specified by pairs of start and end values. These pairs
        /// are specified as adjacent values in an array whose length must therefore be even.
        /// </summary>
        /// <param name="startEndPairs">The array of range starts and ends.</param>
        /// <returns>The created distribution.</returns>
        public static ImmutableDiscreteChar UniformInRanges(params char[] startEndPairs) => UniformInRanges((IEnumerable<char>)startEndPairs);

        /// <summary>
        /// Creates a distribution which is uniform over values in
        /// multiple ranges specified by pairs of start and end values. These pairs
        /// are specified as adjacent values in a sequence whose length must therefore be even.
        /// </summary>
        /// <param name="startEndPairs">The sequence of range starts and ends.</param>
        /// <returns>The created distribution.</returns>
        public static ImmutableDiscreteChar UniformInRanges(IEnumerable<char> startEndPairs) =>
            UniformInRanges(startEndPairs, CharClasses.Unknown, null);

        private static ImmutableDiscreteChar UniformInRanges(IEnumerable<char> startEndPairs, CharClasses charClasses, string regexRepresentation) =>
            new ImmutableDiscreteChar(Storage.CreateUniformInRanges(startEndPairs, charClasses, regexRepresentation));

        /// <summary>
        /// Creates a point mass character distribution.
        /// </summary>
        /// <param name="point">The point.</param>
        /// <returns>The created distribution.</returns>
        public static ImmutableDiscreteChar PointMass(char point) => new ImmutableDiscreteChar(Storage.CreatePoint(point));

        /// <summary>
        /// Creates a point character from a vector of (unnormalized) character probabilities.
        /// </summary>
        /// <param name="vector">The vector of unnormalized probabilities.</param>
        /// <returns>The created distribution.</returns>
        public static ImmutableDiscreteChar FromVector(Vector vector)
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
        public static ImmutableDiscreteChar FromVector(PiecewiseVector vector)
        {
            Argument.CheckIfNotNull(vector, "vector");
            Argument.CheckIfValid(vector.Count <= DiscreteChar.CharRangeEndExclusive, "The given vector is of invalid size.");

            var commonValue = Weight.FromValue(vector.CommonValue);
            int prevEnd = 0;

            var builder = StorageBuilder.Create();
            foreach (var piece in vector.Pieces)
            {
                if (prevEnd != piece.Start && !commonValue.IsZero)
                {
                    builder.AddRange(new CharRange(prevEnd, piece.Start, commonValue));
                }

                builder.AddRange(new CharRange(piece.Start, piece.End + 1, Weight.FromValue(piece.Value)));
                prevEnd = piece.End + 1;
            }

            if (prevEnd < vector.Count && !commonValue.IsZero)
            {
                builder.AddRange(new CharRange(prevEnd, vector.Count, commonValue));
            }

            return new ImmutableDiscreteChar(builder.GetResult());
        }

        #endregion

        #region Distribution interfaces implementation

        /// <inheritdoc cref="Clone"/>
        object ICloneable.Clone() => this;

        /// <summary>
        /// Creates a copy of this distribution.
        /// </summary>
        /// <returns>The created copy.</returns>
        public ImmutableDiscreteChar Clone() => this;

        /// <summary>
        /// Gets the maximum difference between the character probabilities under this distribution and a given one.
        /// </summary>
        /// <param name="distribution">The distribution to compute the maximum probability difference with.</param>
        /// <returns>The computed maximum probability difference.</returns>
        public double MaxDiff(object distribution)
        {
            Argument.CheckIfNotNull(distribution, nameof(distribution));
            return distribution is ImmutableDiscreteChar thatDist
                ? this.Data.MaxDiff(thatDist.Data)
                : double.PositiveInfinity;
        }


        /// <inheritdoc/>
        public ImmutableDiscreteChar CreateUniform() => Uniform();


        /// <inheritdoc/>
        public ImmutableDiscreteChar CreatePointMass(char point) => PointMass(point);

        /// <summary>
        /// Checks whether this distribution is a uniform distribution over all characters.
        /// </summary>
        /// <returns>
        /// <see langword="true"/> if this distribution is a uniform distribution over all possible characters,
        /// <see langword="false"/> otherwise.
        /// </returns>
        public bool IsUniform()
        {
            foreach (var range in this.Ranges)
            {
                if (Math.Abs(range.Probability.LogValue - UniformProb.LogValue) > Eps)
                {
                    return false;
                }
            }

            return true;
        }

        /// <inheritdoc/>
        public double GetLogProb(char value) => FindProb(value).LogValue;

        /// <inheritdoc/>
        public ImmutableDiscreteChar Multiply(ImmutableDiscreteChar other)
        {
            if (IsPointMass)
            {
                if (other.FindProb(Point) == Weight.Zero)
                {
                    throw new AllZeroException("A character distribution that is zero everywhere has been produced.");
                }

                return this;
            }

            if (other.IsPointMass)
            {
                if (FindProb(other.Point) == Weight.Zero)
                {
                    throw new AllZeroException("A character distribution that is zero everywhere has been produced.");
                }

                return other;
            }

            var builder = StorageBuilder.Create();

            foreach (var pair in CharRangePair.IntersectRanges(this, other))
            {
                var probProduct = pair.Probability1 * pair.Probability2;
                builder.AddRange(new CharRange(pair.StartInclusive, pair.EndExclusive, probProduct));
            }

            return new ImmutableDiscreteChar(builder.GetResult());
        }

        /// <summary>
        /// Computes a weighted sum of the current and the given distributions
        /// </summary>
        /// <param name="weightThis">The weight of the current distribution.</param>
        /// <param name="other">The given distribution.</param>
        /// <param name="weightOther">The weight of the given distribution.</param>
        public ImmutableDiscreteChar Sum(double weightThis, ImmutableDiscreteChar other, double weightOther) =>
            Sum(Weight.FromValue(weightThis), other, Weight.FromValue(weightOther));

        /// <summary>
        /// Computes a weighted sum of the current and the given distributions
        /// </summary>
        /// <param name="weightThis">The weight of the current distribution.</param>
        /// <param name="other">The given distribution.</param>
        /// <param name="weightOther">The weight of the given distribution.</param>
        public ImmutableDiscreteChar Sum(Weight weightThis, ImmutableDiscreteChar other, Weight weightOther)
        {
            if (weightThis.IsZero && weightOther.IsZero)
            {
                return Uniform();
            }
            else if (weightThis.IsInfinity)
            {
                if (weightOther.IsInfinity)
                {
                    throw new ArgumentException("both weights are infinity");
                }

                return this;
            }
            else if (weightOther.IsInfinity)
            {
                return other;
            }
            else
            {
                // Make the weights add to 1 to avoid small weights causing a zero mass distribution.
                var invW = Weight.Inverse(weightThis + weightOther);
                weightThis *= invW;
                weightOther *= invW;
                var builder = StorageBuilder.Create();
                foreach (var pair in CharRangePair.CombineRanges(this, other))
                {
                    var probSum = (weightThis * pair.Probability1) + (weightOther * pair.Probability2);
                    builder.AddRange(new CharRange(pair.StartInclusive, pair.EndExclusive, probSum));
                }

                return new ImmutableDiscreteChar(builder.GetResult());
            }
        }
        
        /// <inheritdoc/>
        public double GetLogAverageOf(ImmutableDiscreteChar distribution)
        {
            if (distribution.IsPointMass)
            {
                return this.GetLogProb(distribution.Point);
            }

            if (this.IsPointMass)
            {
                return distribution.GetLogProb(this.Point);
            }

            var result = Weight.Zero;
            foreach (var pair in CharRangePair.IntersectRanges(this, distribution))
            {
                result += Weight.Product(
                    pair.Probability1,
                    pair.Probability2,
                    Weight.FromValue(pair.EndExclusive - pair.StartInclusive));
            }

            return result.LogValue;
        }

        /// <inheritdoc/>
        public ImmutableDiscreteChar CreatePartialUniform()
        {
            var builder = StorageBuilder.Create();
            foreach (var range in Ranges)
            {
                builder.AddRange(
                    new CharRange(
                        range.StartInclusive,
                        range.EndExclusive,
                        range.Probability.IsZero ? Weight.Zero : Weight.One));
            }

            return new ImmutableDiscreteChar(builder.GetResult());
        }

        /// <inheritdoc/>
        public bool IsPartialUniform()
        {
            Weight? commonProb = null;
            foreach (var range in this.Ranges)
            {
                if (commonProb.HasValue && !range.Probability.IsZero && Math.Abs(commonProb.Value.LogValue - range.Probability.LogValue) > Eps)
                {
                    return false;
                }

                commonProb = range.Probability;
            }

            return true;
        }

        /// <inheritdoc/>
        public ImmutableDiscreteChar Divide(ImmutableDiscreteChar denominator, bool forceProper = false)
        {
            var builder = StorageBuilder.Create();

            foreach (var pair in CharRangePair.CombineRanges(this, denominator))
            {
                var probRatio = DivideProb(pair.Probability1, pair.Probability2);
                builder.AddRange(new CharRange(pair.StartInclusive, pair.EndExclusive, probRatio));
            }

            return new ImmutableDiscreteChar(builder.GetResult());
        }

        /// <inheritdoc/>
        public ImmutableDiscreteChar Pow(double power)
        {
            if (power == 0)
            {
                return Uniform();
            }

            var builder = StorageBuilder.Create();

            var prevRangeEnd = 0;
            foreach (var range in Ranges)
            {
                if ((prevRangeEnd != range.StartInclusive || range.Probability.IsZero) && power < 0)
                {
                    throw new DivideByZeroException();
                }

                builder.AddRange(new CharRange(range.StartInclusive, range.EndExclusive, Weight.Pow(range.Probability, power)));
                prevRangeEnd = range.EndExclusive;
            }

            if (prevRangeEnd != DiscreteChar.CharRangeEndExclusive && power < 0)
            {
                throw new DivideByZeroException();
            }

            return new ImmutableDiscreteChar(builder.GetResult());
        }

        /// <inheritdoc/>
        public double GetLogAverageOfPower(ImmutableDiscreteChar distribution, double power)
        {
            // Have to special-case powerof zero because otherwise due to summation in log space being
            // inaccurate, computed result will be on the order of 1e-15 instead of 0.
            if (power < Eps)
            {
                return 0;
            }

            var result = Weight.Zero;
            foreach (var pair in CharRangePair.IntersectRanges(this, distribution))
            {
                if (pair.Probability2.IsZero && power < 0)
                {
                    throw new DivideByZeroException();
                }

                result += Weight.Product(
                    pair.Probability1,
                    Weight.Pow(pair.Probability2, power),
                    Weight.FromValue(pair.EndExclusive - pair.StartInclusive));
            }

            return result.LogValue;
        }

        /// <inheritdoc/>
        public double GetAverageLog(ImmutableDiscreteChar distribution)
        {
            double result = 0;
            foreach (var pair in CharRangePair.CombineRanges(this, distribution))
            {
                var product = ValueTimesLogValue(pair.Probability1, pair.Probability2);
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
            char mode = '\0';
            var maxProb = Weight.Zero;
            foreach (var range in this.Ranges)
            {
                if (range.Probability > maxProb)
                {
                    mode = (char)range.StartInclusive;
                    maxProb = range.Probability;
                }
            }

            return mode;
        }

        /// <inheritdoc/>
        public char Sample()
        {
            var sampleProb = Rand.Double();

            foreach (var interval in this.Ranges)
            {
                var intervalLength = Weight.FromValue(interval.EndExclusive - interval.StartInclusive);
                var prob = intervalLength * interval.Probability;
                sampleProb -= prob.Value;
                if (sampleProb < 0)
                {
                    return (char)(interval.StartInclusive - sampleProb / interval.Probability.Value);
                }
            }

            throw new Exception();
        }

        /// <summary>
        /// Draws a sample from the distribution.
        /// </summary>
        /// <param name="result">A pre-allocated storage for the sample (will be ignored).</param>
        /// <returns>The drawn sample.</returns>
        public char Sample(char result) => this.Sample();

        /// <summary>
        /// Enumerates over the support of the distribution instance.
        /// </summary>
        /// <returns>The character values with non-zero mass.</returns>
        public IEnumerable<char> EnumerateSupport()
        {
            foreach (var range in this.Ranges)
            {
                if (!range.Probability.IsZero)
                {
                    for (int j = range.StartInclusive; j < range.EndExclusive; j++)
                    {
                        yield return (char)j;
                    }
                }
            }
        }
        #endregion

        #region Character distribution specific interface

        /// <summary>
        /// Gets an array of character ranges with associated probabilities.
        /// </summary>
        /// <value>An array of character ranges with associated probabilities.</value>
        public ReadOnlyArray<CharRange> Ranges => this.Data.Ranges;

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
        public ImmutableDiscreteChar Complement() => new ImmutableDiscreteChar(this.Data.Complement());

        public static ImmutableDiscreteChar ToLower(ImmutableDiscreteChar unnormalizedCharDist)
        {
            switch (unnormalizedCharDist.Data.CharClasses)
            {
                case CharClasses.Digit:
                case CharClasses.Lower:
                    return unnormalizedCharDist;
                case CharClasses.Upper:
                case CharClasses.Letter:
                    return Lower();
                case CharClasses.LetterOrDigit:
                    return new ImmutableDiscreteChar(StorageCache.LowerOrDigit);
                case CharClasses.WordChar:
                    return new ImmutableDiscreteChar(StorageCache.LowerWordCharOrDigit);
                case CharClasses.Uniform:
                    return new ImmutableDiscreteChar(StorageCache.UpperComplement);
                default:
                    // TODO: decent implementation
                    var ranges = unnormalizedCharDist.Ranges;
                    var probVector = PiecewiseVector.Zero(DiscreteChar.CharRangeEndExclusive);
                    foreach (var range in ranges)
                    {
                        var rangeWeight = range.Probability;
                        for (var ch = range.StartInclusive; ch < range.EndExclusive; ch++)
                        {
                            var transformedChar = char.ToLowerInvariant((char)ch);
                            probVector[transformedChar] += rangeWeight.Value;
                        }
                    }
                    return FromVector(probVector);
            }
        }

        /// <summary>
        /// Gets a vector of character probabilities under this distribution.
        /// </summary>
        /// <returns>A vector of character probabilities.</returns>
        public PiecewiseVector GetProbs()
        {
            // TODO: replace with GetLogProbs()
            var result = PiecewiseVector.Constant(DiscreteChar.CharRangeEndExclusive, 0);
            foreach (var range in this.Ranges)
            {
                result.Pieces.Add(new ConstantVector(range.StartInclusive, range.EndExclusive - 1, range.Probability.Value));
            }

            return result;
        }

        #endregion

        #region Equals, GetHashCode, ToString

        /// <summary>
        /// Gets a string that represents this distribution.
        /// </summary>
        /// <returns>
        /// A string that represents this distribution.
        /// </returns>
        public override string ToString() => this.Data.ToString();

        public void AppendToString(StringBuilder stringBuilder) => this.Data.AppendToString(stringBuilder);

        /// <summary>
        /// Appends a regex expression that represents this character to the supplied string builder.
        /// </summary>
        /// <param name="stringBuilder">The string builder to append to</param>
        /// <param name="useFriendlySymbols">Whether to use friendly symbols."</param>
        public void AppendRegex(StringBuilder stringBuilder, bool useFriendlySymbols = false) =>
            this.Data.AppendRegex(stringBuilder, useFriendlySymbols);

        /// <summary>
        /// Checks if <paramref name="that"/> equals to this distribution (i.e. represents the same distribution over characters).
        /// </summary>
        /// <param name="that">The object to compare this distribution with.</param>
        /// <returns><see langword="true"/> if this distribution is equal to <paramref name="that"/>, false otherwise.</returns>
        public bool Equals(ImmutableDiscreteChar that) => this.Data.Equals(that.Data);

        /// <summary>
        /// Checks if <paramref name="obj"/> equals to this distribution (i.e. represents the same distribution over characters).
        /// </summary>
        /// <param name="obj">The object to compare this distribution with.</param>
        /// <returns><see langword="true"/> if this distribution is equal to <paramref name="obj"/>, false otherwise.</returns>
        public override bool Equals(object obj) => obj is ImmutableDiscreteChar that && this.Equals(that);

        /// <summary>
        /// Gets the hash code of this distribution.
        /// </summary>
        /// <returns>The hash code.</returns>
        public override int GetHashCode() => this.Data.GetHashCode();

        #endregion

        #region Helper methods

        #region Arithmetic helpers

        /// <summary>
        /// Computes the ratio of two numbers, redefining zero divided by zero to be zero.
        /// </summary>
        /// <param name="numerator">The numerator in the ratio.</param>
        /// <param name="denominator">The denominator in the ratio.</param>
        /// <returns>The ratio.</returns>
        private static Weight DivideProb(Weight numerator, Weight denominator)
        {
            if (denominator.IsZero)
            {
                if (!numerator.IsZero)
                {
                    throw new DivideByZeroException();
                }

                return Weight.Zero;
            }

            return numerator * Weight.Inverse(denominator);
        }

        /// <summary>
        /// Computes <c>value1 * Log(value2)</c>, redefining zero times negative infinity to be zero.
        /// </summary>
        /// <param name="value1">The first value.</param>
        /// <param name="value2">The second value.</param>
        /// <returns>The computed value.</returns>
        private static double ValueTimesLogValue(Weight value1, Weight value2)
        {
            if (value2.IsZero)
            {
                if (!value1.IsZero)
                {
                    return double.NegativeInfinity;
                }

                return 0;
            }

            return value1.Value * value2.LogValue;
        }

        #endregion

        private Weight FindProb(char value)
        {
            foreach (var range in this.Ranges)
            {
                if (range.StartInclusive <= value && range.EndExclusive > value)
                {
                    return range.Probability;
                }
            }

            return Weight.Zero;
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
        public struct CharRange : IComparable<CharRange>
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="CharRange"/> struct
            /// with given range bounds and character probability.
            /// </summary>
            /// <param name="startInclusive">The start of the character range (inclusive).</param>
            /// <param name="endExclusive">The end of the character range (exclusive).</param>
            /// <param name="probability">The probability of characters in the range.</param>
            [Construction("StartInclusive", "EndExclusive", "Probability")]
            public CharRange(int startInclusive, int endExclusive, Weight probability)
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
            /// <remarks>
            /// Setter is required for DataContractSerializer
            /// </remarks>
            [DataMember]
            public int StartInclusive { get; private set; }

            /// <summary>
            /// Gets or sets the end of the range (exclusive).
            /// </summary>
            /// <remarks>
            /// Setter is required for DataContractSerializer
            /// </remarks>
            [DataMember]
            public int EndExclusive { get; private set; }

            /// <summary>
            /// Gets or sets the probability associated with the range.
            /// </summary>
            /// <remarks>
            /// Setter is required for DataContractSerializer
            /// </remarks>
            [DataMember]
            public Weight Probability { get; private set; }

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

            public int CompareTo(CharRange that) =>
                this.StartInclusive.CompareTo(that.StartInclusive);

            internal void AppendToString(StringBuilder stringBuilder)
            {
                stringBuilder.Append('[');
                AppendChar(stringBuilder, (char)this.StartInclusive);

                if (this.StartInclusive == DiscreteChar.CharRangeEndExclusive)
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
                stringBuilder.Append(this.Probability.Value.ToString("G4"));
            }

            /// <summary>
            /// Writes a char range.
            /// </summary>
            public void Write(Action<int> writeInt32, Action<double> writeDouble)
            {
                writeInt32(StartInclusive);
                writeInt32(EndExclusive);
                writeDouble(Probability.LogValue);
            }

            /// <summary>
            /// Reads a char range.
            /// </summary>
            public static CharRange Read(Func<int> readInt32, Func<double> readDouble) => new CharRange(readInt32(), readInt32(), Weight.FromLogValue(readDouble()));
        }

        /// <summary>
        /// Represents a character range with 2 different probabilities attached to it.
        /// </summary>
        private struct CharRangePair
        {
            /// <summary>
            /// Gets or sets the start of the range (inclusive).
            /// </summary>
            public int StartInclusive { get; private set; }

            /// <summary>
            /// Gets or sets the end of the range (exclusive).
            /// </summary>
            public int EndExclusive { get; private set; }

            /// <summary>
            /// Gets or sets the first probability value associated with the range.
            /// </summary>
            public Weight Probability1 { get; private set; }

            /// <summary>
            /// Gets or sets the second probability value associated with the range.
            /// </summary>
            public Weight Probability2 { get; private set; }

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

            private void AppendToString(StringBuilder stringBuilder)
            {
                stringBuilder.Append('[');
                AppendChar(stringBuilder, (char)this.StartInclusive);

                if (this.StartInclusive == DiscreteChar.CharRangeEndExclusive)
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
                stringBuilder.Append(this.Probability1.Value.ToString("G4"));
                stringBuilder.Append(", ");
                stringBuilder.Append(this.Probability2.Value.ToString("G4"));
            }

            /// <summary>
            /// Gets the intersecting ranges of two distributions.
            /// </summary>
            /// <param name="distribution1">The first distribution.</param>
            /// <param name="distribution2">The second distribution</param>
            public static IEnumerable<CharRangePair> IntersectRanges(
                ImmutableDiscreteChar distribution1, ImmutableDiscreteChar distribution2)
            {
                var ranges1 = distribution1.Ranges;
                var ranges2 = distribution2.Ranges;

                var rangeIndex1 = 0;
                var rangeIndex2 = 0;

                while (rangeIndex1 < ranges1.Count && rangeIndex2 < ranges2.Count)
                {
                    var range1 = ranges1[rangeIndex1];
                    var range2 = ranges2[rangeIndex2];
                    if (range1.StartInclusive >= range2.EndExclusive)
                    {
                        ++rangeIndex2;
                    }
                    else if (range2.StartInclusive >= range1.EndExclusive)
                    {
                        ++rangeIndex1;
                    }
                    else
                    {
                        yield return new CharRangePair()
                        {
                            StartInclusive = Math.Max(range1.StartInclusive, range2.StartInclusive),
                            EndExclusive = Math.Min(range1.EndExclusive, range2.EndExclusive),
                            Probability1 = range1.Probability,
                            Probability2 = range2.Probability,
                        };

                        if (range1.EndExclusive <= range2.EndExclusive)
                        {
                            ++rangeIndex1;
                        }

                        if (range2.EndExclusive <= range1.EndExclusive)
                        {
                            ++rangeIndex2;
                        }
                    }
                }
            }

            /// <summary>
            /// Returns all ranges which have non-zero probability in either of disrtibutions.
            /// </summary>
            /// <param name="distribution1">The first distribution.</param>
            /// <param name="distribution2">The second distribution</param>
            public static IEnumerable<CharRangePair> CombineRanges(
                ImmutableDiscreteChar distribution1, ImmutableDiscreteChar distribution2)
            {
                return CombineRanges(distribution1.Data.Ranges, distribution2.Data.Ranges);
            }

            internal static IEnumerable<CharRangePair> CombineRanges(
                ReadOnlyArray<CharRange> ranges1, ReadOnlyArray<CharRange> ranges2)
            {
                var rangeIndex1 = 0;
                var rangeIndex2 = 0;
                var prevEndExclusive = 0;

                while (prevEndExclusive != DiscreteChar.CharRangeEndExclusive)
                {
                    var startInclusive = prevEndExclusive;
                    var endExclusive = DiscreteChar.CharRangeEndExclusive;
                    var probability1 = ProcessRange(
                        ranges1, startInclusive, ref rangeIndex1, ref endExclusive);
                    var probability2 = ProcessRange(
                        ranges2, startInclusive, ref rangeIndex2, ref endExclusive);

                    yield return new CharRangePair()
                    {
                        StartInclusive = startInclusive,
                        EndExclusive = endExclusive,
                        Probability1 = probability1,
                        Probability2 = probability2
                    };

                    prevEndExclusive = endExclusive;
                }

                Weight ProcessRange(
                    ReadOnlyArray<CharRange> ranges,
                    int startInclusive,
                    ref int index,
                    ref int endExclusive)
                {
                    if (index < ranges.Count && ranges[index].EndExclusive == startInclusive)
                    {
                        ++index;
                    }

                    if (index >= ranges.Count)
                    {
                        return Weight.Zero;
                    }

                    if (ranges[index].StartInclusive > startInclusive)
                    {
                        endExclusive = Math.Min(endExclusive, ranges[index].StartInclusive);
                        return Weight.Zero;
                    }

                    endExclusive = Math.Min(endExclusive, ranges[index].EndExclusive);
                    return ranges[index].Probability;
                }
            }
        }

        #endregion

        #region Serialization

        /// <summary>
        /// Writes a discrete character.
        /// </summary>
        public void Write(Action<int> writeInt32, Action<double> writeDouble) =>
            this.Data.Write(writeInt32, writeDouble);

        /// <summary>
        /// Reads a discrete character.
        /// </summary>
        public static ImmutableDiscreteChar Read(Func<int> readInt32, Func<double> readDouble) =>
            new ImmutableDiscreteChar(Storage.Read(readInt32, readDouble));

        /// <summary>
        /// Constructor used during deserialization by Newtonsoft.Json and BinaryFormatter.
        /// </summary>
        private ImmutableDiscreteChar(SerializationInfo info, StreamingContext context) =>
            this.data_ = Storage.FromSerializationInfo(info);

        void ISerializable.GetObjectData(SerializationInfo info, StreamingContext context) =>
            this.Data.GetObjectData(info);

        #endregion

        /// <summary>
        /// Immutable class to hold ImmutableDiscreteChar state
        /// </summary>
        /// <remarks>
        /// This class is serializable but is not marked with <see cref="SerializableAttribute"/> and
        /// <see cref="DataContractAttribute"/> because we have to implement serialization manually
        /// due to Newtonsoft.Json not deserializing <see cref="ReadOnlyArray{T}"/> properly without
        /// "JsonObjectAttribute". Which can't be added because Infer.NET has no explicit dependency
        /// on Newtonsoft.Json.
        /// </remarks>
        internal sealed class Storage
        {
            #region State

            /// <summary>
            /// The array of character ranges with associated probabilities.
            /// </summary>
            /// <remarks>
            /// The character probabilities must be kept normalized by applying <see cref="StorageBuilder.NormalizeProbabilities"/> when necessary.
            /// </remarks>
            public ReadOnlyArray<CharRange> Ranges { get; }

            public char? Point { get; }

            // Following members are not immutable and can be recalculated on-demand
            public CharClasses CharClasses { get; private set; }

            private string regexRepresentation;
            private string symbolRepresentation;

            #endregion

            #region Constructor and factory methods

            private Storage(
                ReadOnlyArray<CharRange> ranges,
                char? point,
                CharClasses charClasses,
                string regexRepresentation,
                string symbolRepresentation)
            {
                this.Ranges = ranges;
                this.Point = point;
                this.CharClasses = charClasses;
                this.regexRepresentation = regexRepresentation;
                this.symbolRepresentation = symbolRepresentation;
            }

            public static Storage CreateUncached(
                ReadOnlyArray<CharRange> ranges,
                char? point,
                CharClasses charClasses = CharClasses.Unknown,
                string regexRepresentation = null,
                string symbolRepresentation = null)
            {
                Debug.Assert(point.HasValue == IsRangesPointMass(ranges));
                return new Storage(ranges, point, charClasses, regexRepresentation, symbolRepresentation);
            }

            public static Storage Create(
                ReadOnlyArray<CharRange> ranges,
                CharClasses charClasses = CharClasses.Unknown,
                string regexRepresentation = null,
                string symbolRepresentation = null)
            {
                return IsRangesPointMass(ranges)
                    ? CreatePoint((char)ranges[0].StartInclusive, ranges)
                    : CreateUncached(ranges, null, charClasses, regexRepresentation, symbolRepresentation);
            }

            public static Storage CreatePoint(char point, ReadOnlyArray<CharRange> ranges) =>
                StorageCache.GetPointMass(point, ranges);

            public static Storage CreatePoint(char point) =>
                StorageCache.GetPointMass(point, null);

            public static Storage CreateUniformInRanges(
                IEnumerable<char> startEndPairs,
                CharClasses charClasses = CharClasses.Unknown,
                string regexRepresentation = null,
                string symbolRepresentation = null)
            {
                Argument.CheckIfNotNull(startEndPairs, "startEndPairs");
                var startEndPairsArray = startEndPairs.ToArray();
                Argument.CheckIfValid(startEndPairsArray.Length % 2 == 0, "startEndPairs", "The number of characters must be even.");

                var builder = StorageBuilder.Create(
                    charClasses, regexRepresentation, symbolRepresentation);
                for (int i = 0; i < startEndPairsArray.Length; i += 2)
                {
                    var startInclusive = startEndPairsArray[i];
                    var endExclusive = startEndPairsArray[i + 1] + 1;
                    if (startInclusive >= endExclusive)
                        throw new ArgumentException("Inverted character range");

                    builder.AddRange(new CharRange(startInclusive, endExclusive, Weight.One));
                }

                builder.SortAndCheckRanges();
                return builder.GetResult();
            }

            #endregion

            #region Helper methods

            public Storage Complement()
            {
                // Must use StorageBuilder, because need to Normalize probabilities
                var builder = StorageBuilder.Create();
                int prevEnd = 0;
                foreach (var range in this.Ranges)
                {
                    if (range.StartInclusive != prevEnd)
                    {
                        builder.AddRange(new CharRange(prevEnd, range.StartInclusive, Weight.One));
                    }

                    prevEnd = range.EndExclusive;
                }

                if (prevEnd != DiscreteChar.CharRangeEndExclusive)
                {
                    builder.AddRange(new CharRange(prevEnd, DiscreteChar.CharRangeEndExclusive, Weight.One));
                }

                return builder.GetResult();
            }

            public double MaxDiff(Storage other)
            {
                double result = 0;
                foreach (var pair in CharRangePair.CombineRanges(this.Ranges, other.Ranges))
                {
                    result = Math.Max(result, Weight.AbsoluteDifference(pair.Probability1, pair.Probability2).Value);
                }

                return result;
            }

            public bool Equals(Storage that) =>
                this.Point.HasValue
                    ? that.Point.HasValue && this.Point.Value == that.Point.Value
                    : this.CharClasses != CharClasses.Unknown && this.CharClasses == that.CharClasses || this.MaxDiff(that) < Eps;

            public override bool Equals(object obj) => obj is Storage that && this.Equals(that);

            // TODO: What to do here? every distribution has multiple representations
            public override int GetHashCode() => 17;

            #endregion

            #region Properties

            // TODO: Assumes that there are no ranges with zero probability
            private static bool IsRangesPointMass(ReadOnlyArray<CharRange> ranges) =>
                ranges.Count > 0 && Math.Abs(ranges[0].Probability.LogValue - Weight.One.LogValue) < Eps;

            /// <summary>
            /// Returns weather char class of this state ImmutableDiscreteChar equals charClass.
            /// If current char class is unknown it is first updated using comparison with instance of ImmutableDiscreteChar which belongs to this class.
            /// </summary>
            private bool IsCharClass(CharClasses charClass, Func<ImmutableDiscreteChar> classConstructor)
            {
                if (CharClasses == CharClasses.Unknown && this.Equals(classConstructor().Data))
                {
                    this.CharClasses = charClass;
                }

                return this.CharClasses == charClass;
            }

            public bool IsDigit => this.IsCharClass(CharClasses.Digit, Digit);

            public bool IsLower => this.IsCharClass(CharClasses.Lower, Lower);

            public bool IsUpper => this.IsCharClass(CharClasses.Upper, Upper);

            public bool IsLetter => this.IsCharClass(CharClasses.Letter, Letter);

            public bool IsLetterOrDigit => this.IsCharClass(CharClasses.LetterOrDigit, LetterOrDigit);

            public bool IsWordChar => this.IsCharClass(CharClasses.WordChar, WordChar);

            #endregion

            #region Serialization

            public static Storage FromSerializationInfo(SerializationInfo info)
            {
                var ranges = (CharRange[]) info.GetValue(nameof(Ranges), typeof(CharRange[]));
                var classes = (CharClasses) info.GetValue(nameof(CharClasses), typeof(CharClasses));
                return Storage.Create(ranges.ToReadOnlyArray(), classes);
            }

            public void GetObjectData(SerializationInfo info)
            {
                info.AddValue(nameof(this.Ranges), this.Ranges.CloneArray());
                info.AddValue(nameof(this.CharClasses), this.CharClasses);
            }

            public void Write(Action<int> writeInt32, Action<double> writeDouble)
            {
                writeInt32(this.Ranges.Count);
                foreach (var range in this.Ranges)
                {
                    range.Write(writeInt32, writeDouble);
                }
                writeInt32((int)this.CharClasses);
            }

            /// <summary>
            /// Reads a discrete character.
            /// </summary>
            public static Storage Read(Func<int> readInt32, Func<double> readDouble)
            {
                var nRanges = readInt32();
                var ranges = new CharRange[nRanges];
                for (var i = 0; i < nRanges; i++)
                {
                    ranges[i] = CharRange.Read(readInt32, readDouble);
                }

                var charClasses = (CharClasses)readInt32();

                return Storage.Create(ranges.ToReadOnlyArray(), charClasses);
            }

            #endregion

            #region ToString, RegexRepresentation, SymbolRepresentation

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
                if (this.Point.HasValue)
                {
                    AppendChar(stringBuilder, this.Point.Value);
                }
                else
                {
                    foreach (var range in this.Ranges)
                    {
                        range.AppendToString(stringBuilder);
                        stringBuilder.Append(' ');
                    }
                }
            }

            /// <summary>
            /// Appends a regex expression that represents this character to the supplied string builder.
            /// </summary>
            /// <param name="stringBuilder">The string builder to append to</param>
            /// <param name="useFriendlySymbols">Whether to use friendly symbols."</param>
            public void AppendRegex(StringBuilder stringBuilder, bool useFriendlySymbols = false) =>
                stringBuilder.Append(useFriendlySymbols ? this.SymbolRepresentation : this.RegexRepresentation);

            public string RegexRepresentation
            {
                get
                {
                    if (this.regexRepresentation == null)
                    {
                        if (this.Point.HasValue)
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
                        // TODO: handle uniform
                        else
                        {
                            var representation = new StringBuilder();
                            this.AppendRanges(representation, true);
                            this.regexRepresentation = representation.ToString();
                        }
                    }

                    return regexRepresentation;
                }
            }

            public string SymbolRepresentation
            {
                get
                {
                    if (this.symbolRepresentation == null)
                    {
                        if (this.Point.HasValue)
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
                        // TODO: handle uniform
                        else
                        {
                            var representation = new StringBuilder();
                            this.AppendRanges(representation, false);
                            this.symbolRepresentation = representation.ToString();
                        }
                    }

                    return symbolRepresentation;
                }
            }

            private void AppendRanges(StringBuilder representation, bool escape)
            {
                if (this.Ranges.Count >= 1)
                {
                    representation.Append('[');
                    foreach (var range in this.Ranges)
                    {
                        AppendCharacterRange(representation, range, escape);
                    }

                    representation.Append(']');
                }
            }

            #endregion
        }

        /// <summary>
        /// Internal cache for the most used character distributions. Creates them lazily on demand
        /// </summary>
        internal sealed class StorageCache
        {
            // Common distributions which can be created using factory methods
            public static readonly Storage Uniform;
            public static readonly Storage Digit;
            public static readonly Storage Lower;
            public static readonly Storage Upper;
            public static readonly Storage Letter;
            public static readonly Storage LetterOrDigit;
            public static readonly Storage WordChar;
            public static readonly Storage NonWordChar;
            public static readonly Storage Whitespace;

            // Common distributions which can be produced by ToLower()
            public static readonly Storage LowerOrDigit;
            public static readonly Storage LowerWordCharOrDigit;
            public static readonly Storage UpperComplement;

            private static readonly Storage[] PointMasses;

            static StorageCache()
            {
                string LetterOrDigitsRanges(string baseRange) => baseRange + "09";
                string WordCharRanges(string baseRange) => baseRange + "09__";

                Uniform = Storage.CreateUncached(
                    ReadOnlyArray.Create(new CharRange(char.MinValue, DiscreteChar.CharRangeEndExclusive, UniformProb)),
                    null,
                    CharClasses.Uniform,
                    UniformRegexRepresentation,
                    UniformSymbolRepresentation);
                Digit = Storage.CreateUniformInRanges("09", CharClasses.Digit);
                Lower = Storage.CreateUniformInRanges(DiscreteChar.LowerCaseCharacterRanges, CharClasses.Lower);
                Upper = Storage.CreateUniformInRanges(DiscreteChar.UpperCaseCharacterRanges, CharClasses.Upper);
                Letter = Storage.CreateUniformInRanges(DiscreteChar.LetterCharacterRanges, CharClasses.Letter);
                LetterOrDigit = Storage.CreateUniformInRanges(LetterOrDigitsRanges(DiscreteChar.LetterCharacterRanges), CharClasses.LetterOrDigit);
                WordChar = Storage.CreateUniformInRanges(WordCharRanges(DiscreteChar.LetterCharacterRanges), CharClasses.WordChar);
                NonWordChar = WordChar.Complement();
                Whitespace = Storage.CreateUniformInRanges("\t\r  ");

                LowerOrDigit = Storage.CreateUniformInRanges(
                    LetterOrDigitsRanges(DiscreteChar.LowerCaseCharacterRanges),
                    regexRepresentation: @"[\p{Ll}\d]",
                    symbolRepresentation: "⭽");
                LowerWordCharOrDigit = Storage.CreateUniformInRanges(
                    WordCharRanges(DiscreteChar.LowerCaseCharacterRanges),
                    regexRepresentation: @"[\p{Ll}\d_]",
                    symbolRepresentation: "⧬");

                var upperComplement = Upper.Complement();
                UpperComplement = Storage.CreateUncached(
                    upperComplement.Ranges,
                    null,
                    regexRepresentation: @"[^\p{Lu}]",
                    symbolRepresentation: "🡻");

                PointMasses = new Storage[DiscreteChar.CharRangeEndExclusive];
            }

            public static Storage GetPointMass(char point, ReadOnlyArray<CharRange>? ranges)
            {
                if (PointMasses[point] == null)
                {
                    PointMasses[point] = Storage.CreateUncached(
                        ranges.HasValue
                            ? ranges.Value
                            : ReadOnlyArray.Create(new CharRange(point, point + 1, Weight.One)),
                        point);
                }

                return PointMasses[point];
            }
        }

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
        /// Helper struct for building Storage using builder pattern.
        /// </summary>
        /// <remarks>
        /// It is a mutable struct instead of class for performance reasons - we want ot avoid
        /// allocations which aren't necessary. StorageBuilder is always created on stack and
        /// not shared with outer code. Which means we don't get to experience drawbacks
        /// of mutable structs.
        /// </remarks>
        private struct StorageBuilder
        {

            #region State

            /// <summary>
            /// The array of character ranges with associated probabilities.
            /// </summary>
            private readonly List<CharRange> ranges;

            /// <summary>
            /// Precomputed character class.
            /// </summary>
            private readonly CharClasses charClasses;

            /// <summary>
            /// Precomputed regex representation.
            /// </summary>
            private string regexRepresentation;

            /// <summary>
            /// Precomputed symbol representation.
            /// </summary>
            private readonly string symbolRepresentation;

            #endregion

            public StorageBuilder(
                CharClasses charClasses,
                string regexRepresentation,
                string symbolRepresentation)
            {
                this.ranges = new List<CharRange>();
                this.charClasses = charClasses;
                this.regexRepresentation = regexRepresentation;
                this.symbolRepresentation = symbolRepresentation;
            }

            public static StorageBuilder Create(
                CharClasses charClasses = CharClasses.Unknown,
                string regexRepresentation = null,
                string symbolRepresentation = null)
            {
                return new StorageBuilder(charClasses, regexRepresentation, symbolRepresentation);
            }

            #region Public methods

            /// <summary>
            /// Adds a new character range to <see cref="ranges"/>
            /// </summary>
            public void AddRange(CharRange range)
            {
                if (!range.Probability.IsZero)
                {
                    this.ranges.Add(range);
                }
            }

            /// <summary>
            /// Sorts ranges by StartInclusive, checks that they are non-overlapping, cover valid characters only
            /// and have non-negative (unnormalized) probability values associated with them.
            /// </summary>
            /// <remarks>
            /// For performance reasons calling this method is optional if ranges were added in proper order.
            /// </remarks>
            public void SortAndCheckRanges()
            {
                Debug.Assert(this.ranges.Count > 0);

                this.ranges.Sort();

                var prevRangeEnd = 0;
                foreach (var range in this.ranges)
                {
                    Argument.CheckIfValid(
                        range.StartInclusive >= prevRangeEnd && range.EndExclusive <= DiscreteChar.CharRangeEndExclusive,
                        nameof(this.ranges),
                        "Ranges must only include valid characters and be non-overlapping.");
                    prevRangeEnd = range.EndExclusive;
                }
            }

            /// <summary>
            /// Normalizes probabilities in ranges and returns build Storage.
            /// </summary>
            public Storage GetResult()
            {
                this.MergeNeighboringRanges();
                NormalizeProbabilities(this.ranges);
                return Storage.Create(
                    this.ranges.ToReadOnlyArray(),
                    this.charClasses,
                    this.regexRepresentation,
                    this.symbolRepresentation);
            }

            #endregion

            #region Helper methods

            /// <summary>
            /// Merges neighboring character ranges that have the same associated probability value.
            /// </summary>
            private void MergeNeighboringRanges()
            {
                var newRangeCount = 0;
                for (var i = 0; i < this.ranges.Count; ++i)
                {
                    var range = this.ranges[i];

                    if (newRangeCount > 0)
                    {
                        var prevRange = this.ranges[newRangeCount - 1];
                        if (range.StartInclusive == prevRange.EndExclusive &&
                            Math.Abs(range.Probability.LogValue - prevRange.Probability.LogValue) < Eps)
                        {
                            this.ranges[newRangeCount - 1] = new CharRange(
                                prevRange.StartInclusive, range.EndExclusive, prevRange.Probability);
                            continue;
                        }
                    }

                    this.ranges[newRangeCount++] = range;
                }

                this.ranges.RemoveRange(newRangeCount, this.ranges.Count - newRangeCount);
            }

            /// <summary>
            /// Normalizes probabilities in ranges.
            /// </summary>
            /// <param name="ranges">The ranges.</param>
            /// <exception cref="ArgumentException">Thrown if logProbabilityOverride has value corresponding to a non-probability.</exception>
            public static void NormalizeProbabilities(IList<CharRange> ranges)
            {
                var normalizer = ComputeInvNormalizer(ranges);
                for (var i = 0; i < ranges.Count; ++i)
                {
                    var range = ranges[i];
                    var probability = range.Probability * normalizer;

                    ranges[i] = new CharRange(
                        range.StartInclusive, range.EndExclusive, probability);
                }
            }

            /// <summary>
            /// Computes the normalizer of this distribution.
            /// </summary>
            /// <returns>The computed normalizer.</returns>
            private static Weight ComputeInvNormalizer(IEnumerable<CharRange> ranges)
            {
                var normalizer = Weight.Zero;

                foreach (var range in ranges)
                {
                    normalizer += Weight.FromValue(range.EndExclusive - range.StartInclusive) * range.Probability;
                }

                if (normalizer.IsZero)
                {
                    throw new AllZeroException("A character distribution that is zero everywhere has been produced.");
                }

                return Weight.Inverse(normalizer);
            }

            #endregion
        }
    }
}
