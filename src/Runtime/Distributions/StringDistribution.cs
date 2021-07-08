// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Runtime.Serialization;

    using Automata;
    using Utilities;
    using Factors.Attributes;

    /// <summary>
    /// Represents a distribution over strings that uses a weighted finite state automaton as the underlying weight function.
    /// </summary>
    [Serializable]
    [DataContract]
    [Quality(QualityBand.Preview)]
    public class StringDistribution :
        SequenceDistribution<
            string,
            char,
            ImmutableDiscreteChar,
            StringManipulator,
            StringAutomaton,
            WeightFunctions<string, char, ImmutableDiscreteChar, StringManipulator, StringAutomaton>.MultiRepresentationWeightFunction<StringDictionaryWeightFunction>,
            WeightFunctions<string, char, ImmutableDiscreteChar, StringManipulator, StringAutomaton>.MultiRepresentationWeightFunction<StringDictionaryWeightFunction>.Factory,
            StringDistribution>
    {
        /// <summary>
        /// Concatenates the weighted regular languages defined by given distributions
        /// (see <see cref="SequenceDistribution{TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TWeightFunction, TWeightFunctionFactory, TThis}.Append(TThis, int)"/>).
        /// </summary>
        /// <param name="first">The first distribution.</param>
        /// <param name="second">The second distribution.</param>
        /// <returns>The concatenation result.</returns>
        public static StringDistribution operator +(StringDistribution first, StringDistribution second)
        {
            return first.Append(second);
        }

        /// <summary>
        /// Creates a point mass distribution.
        /// This method is an alias for <see cref="SequenceDistribution{TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TWeightFunction, TWeightFunctionFactory, TThis}.PointMass"/>.
        /// </summary>
        /// <param name="str">The point.</param>
        /// <returns>The created point mass distribution.</returns>
        public static StringDistribution String(string str)
        {
            return StringDistribution.PointMass(str);
        }

        /// <summary>
        /// Creates a distribution which puts all mass on a string containing only a given character.
        /// This method is an alias for <see cref="SequenceDistribution{TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TWeightFunction, TWeightFunctionFactory, TThis}.SingleElement(TElement)"/>.
        /// </summary>
        /// <param name="ch">The character.</param>
        /// <returns>The created distribution.</returns>
        public static StringDistribution Char(char ch)
        {
            return StringDistribution.SingleElement(ch);
        }

        /// <summary>
        /// Creates a distribution over strings of length 1 induced by a given distribution over characters.
        /// This method is an alias for <see cref="SequenceDistribution{TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TWeightFunction, TWeightFunctionFactory, TThis}.SingleElement(TElementDistribution)"/>.
        /// </summary>
        /// <param name="characterDist">The distribution over characters.</param>
        /// <returns>The created distribution.</returns>
        /// <remarks>
        /// The distribution created by this method can differ from the result of
        /// <see cref="SequenceDistribution{TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TWeightFunction, TWeightFunctionFactory, TThis}.Repeat(TThis, int, int?)"/>
        /// with both min and max length set to 1 since the latter always creates a partial uniform distribution.
        /// </remarks>
        public static StringDistribution Char(ImmutableDiscreteChar characterDist)
        {
            return StringDistribution.SingleElement(characterDist);
        }

        /// <summary>
        /// Creates a distribution over strings of length 1 induced by a given distribution over characters.
        /// This method is an alias for <see cref="SequenceDistribution{TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TWeightFunction, TWeightFunctionFactory, TThis}.SingleElement(TElementDistribution)"/>.
        /// </summary>
        /// <param name="characterDist">The distribution over characters.</param>
        /// <returns>The created distribution.</returns>
        /// <remarks>
        /// The distribution created by this method can differ from the result of
        /// <see cref="SequenceDistribution{TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TWeightFunction, TWeightFunctionFactory, TThis}.Repeat(TThis, int, int?)"/>
        /// with both min and max length set to 1 since the latter always creates a partial uniform distribution.
        /// </remarks>
        public static StringDistribution Char(DiscreteChar characterDist)
        {
            return StringDistribution.SingleElement(characterDist.WrappedDistribution);
        }

        /// <summary>
        /// Creates a uniform distribution over all strings that are case-invariant matches of the specified string.
        /// </summary>
        /// <param name="template">The string to match.</param>
        /// <returns>The created distribution.</returns>
        public static StringDistribution CaseInvariant(string template) =>
            StringDistribution.Concatenate(
                template.Select(
                    ch =>
                    {
                        var upper = char.ToUpperInvariant(ch);
                        var lower = char.ToLowerInvariant(ch);
                        return
                            upper == lower
                                ? ImmutableDiscreteChar.PointMass(lower)
                                : ImmutableDiscreteChar.OneOf(lower, upper);
                    }));

        /// <summary>
        /// Creates a uniform distribution over strings of lowercase letters, with length within the given bounds.
        /// If <paramref name="maxLength"/> is set to <see langword="null"/>,
        /// there will be no upper bound on the length, and the resulting distribution will thus be improper.
        /// </summary>
        /// <param name="minLength">The minimum possible string length. Defaults to 1.</param>
        /// <param name="maxLength">
        /// The maximum possible sequence length, or <see langword="null"/> for no upper bound on length.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static StringDistribution Lower(int minLength = 1, int? maxLength = null)
        {
            return StringDistribution.Repeat(ImmutableDiscreteChar.Lower(), minLength, maxLength);
        }

        /// <summary>
        /// Creates a uniform distribution over strings of uppercase letters, with length within the given bounds.
        /// If <paramref name="maxLength"/> is set to <see langword="null"/>,
        /// there will be no upper bound on the length, and the resulting distribution will thus be improper.
        /// </summary>
        /// <param name="minLength">The minimum possible string length. Defaults to 1.</param>
        /// <param name="maxLength">
        /// The maximum possible sequence length, or <see langword="null"/> for no upper bound on length.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static StringDistribution Upper(int minLength = 1, int? maxLength = null)
        {
            return StringDistribution.Repeat(ImmutableDiscreteChar.Upper(), minLength, maxLength);
        }

        /// <summary>
        /// Creates a uniform distribution over strings of lowercase and uppercase letters, with length in given bounds.
        /// If <paramref name="maxLength"/> is set to <see langword="null"/>,
        /// there will be no upper bound on the length, and the resulting distribution will thus be improper.
        /// </summary>
        /// <param name="minLength">The minimum possible string length. Defaults to 1.</param>
        /// <param name="maxLength">
        /// The maximum possible sequence length, or <see langword="null"/> for no upper bound on length.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static StringDistribution Letters(int minLength = 1, int? maxLength = null)
        {
            return StringDistribution.Repeat(ImmutableDiscreteChar.Letter(), minLength, maxLength);
        }

        /// <summary>
        /// Creates a uniform distribution over strings of digits, with length within the given bounds.
        /// If <paramref name="maxLength"/> is set to <see langword="null"/>,
        /// there will be no upper bound on the length, and the resulting distribution will thus be improper.
        /// </summary>
        /// <param name="minLength">The minimum possible string length. Defaults to 1.</param>
        /// <param name="maxLength">
        /// The maximum possible sequence length, or <see langword="null"/> for no upper bound on length.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <param name="uniformity">The type of uniformity. Defaults to <see cref="DistributionKind.UniformOverValue"/></param>
        /// <returns>The created distribution.</returns>
        public static StringDistribution Digits(int minLength = 1, int? maxLength = null, DistributionKind uniformity = DistributionKind.UniformOverValue)
        {
            return StringDistribution.Repeat(ImmutableDiscreteChar.Digit(), minLength, maxLength, uniformity);
        }

        /// <summary>
        /// Creates a uniform distribution over strings of digits, lowercase and uppercase letters, with length within the given bounds.
        /// If <paramref name="maxLength"/> is set to <see langword="null"/>,
        /// there will be no upper bound on the length, and the resulting distribution will thus be improper.
        /// </summary>
        /// <param name="minLength">The minimum possible string length. Defaults to 1.</param>
        /// <param name="maxLength">
        /// The maximum possible sequence length, or <see langword="null"/> for no upper bound on length.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static StringDistribution LettersOrDigits(int minLength = 1, int? maxLength = null)
        {
            return StringDistribution.Repeat(ImmutableDiscreteChar.LetterOrDigit(), minLength, maxLength);
        }

        /// <summary>
        /// Creates a uniform distribution over strings of word characters (see <see cref="DiscreteChar.WordChar"/>),
        /// with length within the given bounds.
        /// If <paramref name="maxLength"/> is set to <see langword="null"/>,
        /// there will be no upper bound on the length, and the resulting distribution will thus be improper.
        /// </summary>
        /// <param name="minLength">The minimum possible string length. Defaults to 1.</param>
        /// <param name="maxLength">
        /// The maximum possible sequence length, or <see langword="null"/> for no upper bound on length.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static StringDistribution WordChars(int minLength = 1, int? maxLength = null)
        {
            return StringDistribution.Repeat(ImmutableDiscreteChar.WordChar(), minLength, maxLength);
        }

        /// <summary>
        /// Creates a uniform distribution over strings of whitespace characters (see <see cref="DiscreteChar.Whitespace"/>),
        /// with length within the given bounds.
        /// If <paramref name="maxLength"/> is set to <see langword="null"/>,
        /// there will be no upper bound on the length, and the resulting distribution will thus be improper.
        /// </summary>
        /// <param name="minLength">The minimum possible string length. Defaults to 1.</param>
        /// <param name="maxLength">
        /// The maximum possible sequence length, or <see langword="null"/> for no upper bound on length.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static StringDistribution Whitespace(int minLength = 1, int? maxLength = null)
        {
            return StringDistribution.Repeat(ImmutableDiscreteChar.Whitespace(), minLength, maxLength);
        }

        /// <summary>
        /// Creates a uniform distribution over strings that start with an upper case letter followed by
        /// one or more letters, with length within the given bounds.
        /// If <paramref name="maxLength"/> is set to <see langword="null"/>,
        /// there will be no upper bound on the length, and the resulting distribution will thus be improper.
        /// </summary>
        /// <param name="minLength">The minimum possible string length. Defaults to 2.</param>
        /// <param name="maxLength">
        /// The maximum possible sequence length, or <see langword="null"/> for no upper bound on length.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <param name="allowUpperAfterFirst">Whether to allow upper case letters after the initial upper case letter.  If false, only lower case letters will be allowed.</param>
        /// <returns>The created distribution.</returns>
        public static StringDistribution Capitalized(int minLength = 2, int? maxLength = null, bool allowUpperAfterFirst = false)
        {
            Argument.CheckIfInRange(minLength >= 1, "minLength", "The minimum length of a capitalized string should be 1 or more.");
            Argument.CheckIfValid(!maxLength.HasValue || maxLength.Value >= minLength, "The maximum length cannot be less than the minimum length.");

            var result = StringDistribution.Char(ImmutableDiscreteChar.Upper());
            
            if (maxLength.HasValue)
            {
                result.AppendInPlace(
                    allowUpperAfterFirst ? StringDistribution.Letters(minLength: minLength - 1, maxLength: maxLength.Value - 1)
                    : StringDistribution.Lower(minLength: minLength - 1, maxLength: maxLength.Value - 1));
            }
            else
            {
                // Concatenation with an improper distribution, need to adjust its scale so that the result is 1 on its support
                double logNormalizer = result.GetLogAverageOf(result);
                var lowercaseSuffixFunc = (allowUpperAfterFirst ? StringDistribution.Letters(minLength: minLength - 1) 
                    : StringDistribution.Lower(minLength: minLength - 1)).ToNormalizedAutomaton();
                var lowercaseSuffixFuncScaled = lowercaseSuffixFunc.ScaleLog(-logNormalizer);
                result.AppendInPlace(StringDistribution.FromWeightFunction(lowercaseSuffixFuncScaled));    
            }

            return result;
        }

        /// <summary>
        /// Replaces the current distribution by a distribution over concatenations of sequences
        /// from the current distribution and a given element.
        /// </summary>
        /// <param name="element">The element to append.</param>
        /// <param name="group">The group for the appended element.</param>
        /// <remarks>
        /// The result is equivalent to the distribution produced by the following sampling procedure:
        /// <list type="number">
        /// <item><description>
        /// Sample a random sequence from the current distribution.
        /// </description></item>
        /// <item><description>
        /// Append the given element to the sampled sequence and output the result.
        /// </description></item>
        /// </list>
        /// </remarks>
        public void AppendInPlace(DiscreteChar element, int group = 0)
        {
            this.AppendInPlace(SingleElement(element.WrappedDistribution), group);
        }

        /// <summary>
        /// Creates a regex pattern for the current string distribution. This can then in turn be used with the normal Regex .net
        /// class.
        /// </summary>
        /// <returns>A regex pattern.</returns>
        public string ToRegex()
        {
            var sequenceDistributionFormat = new SequenceDistributionFormatPointMassAsString(AutomatonFormats.Regexp);
            var regexStr = this.ToString(sequenceDistributionFormat);
            return regexStr;
        }

        /// <summary>
        /// Creates a string representation of the current instance.
        /// </summary>
        /// <returns>A string representation of the current instance.</returns>
        public override string ToString()
        {
            return base.ToString((eltDist, stringBuilder) => eltDist.AppendRegex(stringBuilder, true));
        }
    }
}
