// <copyright file="WordStrings.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Linq;

    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Distributions.Automata;

    /// <summary>
    /// Sets of strings of whole words.
    /// </summary>
    public static class WordStrings
    {
        /// <summary>
        /// A uniform distribution over all non-word characters.
        /// </summary>
        private static readonly DiscreteChar NonWordCharacter = DiscreteChar.InRanges(DiscreteChar.LetterCharacterRanges + "09__{{}}''").Complement();

        /// <summary>
        /// Creates a uniform distribution over either the empty string or any string ending with a non-word character.
        /// </summary>
        /// <returns>The created distribution.</returns>
        public static StringDistribution WordPrefix() => WordPrefix(DiscreteChar.Any());

        /// <summary>
        /// Creates a uniform distribution over either the empty string or any string ending with a non-word character.
        /// Characters other than the last are restricted to be non-zero probability characters from a given distribution.
        /// </summary>
        /// <param name="allowedChars">The distribution representing allowed characters.</param>
        /// <returns>The created distribution.</returns>
        public static StringDistribution WordPrefix(DiscreteChar allowedChars) => EmptyOrEndsWith(allowedChars, NonWordCharacter);

        public static StringDistribution EmptyOrEndsWith(DiscreteChar charsInMainString, DiscreteChar endsWith)
        {
            // TODO: fix equality and then use factory methods to create this
            var result = new StringAutomaton.Builder();
            result.Start.SetEndWeight(Weight.One);
            var otherState1 = result.Start.AddEpsilonTransition(Weight.One);
            otherState1.AddSelfTransition(charsInMainString, Weight.FromLogValue(-charsInMainString.GetLogAverageOf(charsInMainString)));
            var otherState2 = otherState1.AddTransition(endsWith, Weight.FromLogValue(-endsWith.GetLogAverageOf(endsWith)));
            otherState2.SetEndWeight(Weight.One);

            return StringDistribution.FromWeightFunction(result.GetAutomaton());
        }

        /// <summary>
        /// Creates a uniform distribution over either the empty string or any string starting with a non-word character.
        /// </summary>
        /// <returns>The created distribution.</returns>
        public static StringDistribution WordSuffix() => WordSuffix(DiscreteChar.Any());

        /// <summary>
        /// Creates a uniform distribution over either the empty string or any string starting with a non-word character.
        /// Characters other than the last are restricted to be non-zero probability characters from a given distribution.
        /// </summary>
        /// <param name="allowedChars">The distribution representing allowed characters.</param>
        /// <returns>The created distribution.</returns>
        public static StringDistribution WordSuffix(DiscreteChar allowedChars) => EmptyOrStartsWith(allowedChars, NonWordCharacter);

        public static StringDistribution EmptyOrStartsWith(DiscreteChar charsInMainString, DiscreteChar startsWith)
        {
            // TODO: fix equality and then use factory methods to create this
            var result = new StringAutomaton.Builder();
            result.Start.SetEndWeight(Weight.One);
            var otherState = result.Start.AddTransition(startsWith, Weight.FromLogValue(-startsWith.GetLogAverageOf(startsWith)));
            otherState.AddSelfTransition(charsInMainString, Weight.FromLogValue(-charsInMainString.GetLogAverageOf(charsInMainString)));
            otherState.SetEndWeight(Weight.One);

            return StringDistribution.FromWeightFunction(result.GetAutomaton());
        }

        /// <summary>
        /// Creates a uniform distribution over any string starting and ending with a non-word character (possibly of length 1).
        /// </summary>
        /// <returns>The created distribution.</returns>
        public static StringDistribution WordMiddle() => WordMiddle(DiscreteChar.Any());

        /// <summary>
        /// Creates a uniform distribution over any string starting and ending with a non-word character.
        /// Characters other than the first and the last are restricted to be non-zero probability characters
        /// from a given distribution.
        /// </summary>
        /// <param name="allowedChars">The distribution representing allowed characters.</param>
        /// <param name="nonWordCharacter">The word separating characters.</param>
        /// <returns>The created distribution.</returns>
        public static StringDistribution WordMiddle(DiscreteChar allowedChars, DiscreteChar? nonWordCharacter = null)
        {
            // TODO: fix equality and then use factory methods to create this
            nonWordCharacter = nonWordCharacter ?? NonWordCharacter;
            var result = new StringAutomaton.Builder();
            var otherState1 = result.Start.AddTransition(
                Option.FromNullable(nonWordCharacter),
                Weight.FromLogValue(-nonWordCharacter.Value.GetLogAverageOf(nonWordCharacter.Value)));
            otherState1.SetEndWeight(Weight.One);
            var otherState2 = otherState1.AddEpsilonTransition(Weight.One)
                .AddSelfTransition(allowedChars, Weight.FromLogValue(-allowedChars.GetLogAverageOf(allowedChars))).AddTransition(
                    Option.FromNullable(nonWordCharacter),
                    Weight.FromLogValue(-nonWordCharacter.Value.GetLogAverageOf(nonWordCharacter.Value)));
            otherState2.SetEndWeight(Weight.One);

            return StringDistribution.FromWeightFunction(result.GetAutomaton());
        }

        /// <summary>
        /// Creates a uniform distribution over any string starting and ending with a non-word character,
        /// with a length in given bounds.
        /// </summary>
        /// <param name="minLength">The minimum allowed string length.</param>
        /// <param name="maxLength">The maximum allowed string length.</param>
        /// <returns>The created distribution.</returns>
        public static StringDistribution WordMiddle(int minLength, int maxLength) => WordMiddle(minLength, maxLength, DiscreteChar.Any());

        /// <summary>
        /// Creates a uniform distribution over any string starting and ending with a non-word character,
        /// with a length in given bounds.
        /// Characters other than the first and the last are restricted to be non-zero probability characters
        /// from a given distribution.
        /// </summary>
        /// <param name="minLength">The minimum allowed string length.</param>
        /// <param name="maxLength">The maximum allowed string length.</param>
        /// <param name="allowedChars">The distribution representing allowed characters.</param>
        /// <returns>The created distribution.</returns>
        public static StringDistribution WordMiddle(int minLength, int maxLength, DiscreteChar allowedChars)
        {
            if (maxLength < minLength)
            {
                throw new ArgumentException("The maximum length cannot be less than the minimum length.");
            }

            if (minLength < 1)
            {
                throw new ArgumentException("The minimum length must be at least one.");
            }

            var nonWordChar = StringDistribution.Char(NonWordCharacter);
            if ((minLength == 1) && (maxLength == 1))
            {
                return nonWordChar;
            }

            // TODO: make a PartialUniform copy of allowedChars
            var suffix = StringDistribution.Repeat(allowedChars, minTimes: Math.Max(minLength - 2, 0), maxTimes: maxLength - 2);
            suffix.AppendInPlace(nonWordChar);

            if (minLength == 1)
            {
                var allowedChar = allowedChars.GetMode();
                var allowedSuffix = new string(Enumerable.Repeat(allowedChar, Math.Max(minLength - 2, 0)).ToArray()) + ' ';
                var suffixLogProb = suffix.GetLogProb(allowedSuffix);
                suffix.SetToSumLog(suffixLogProb, StringDistribution.Empty(), 0.0, suffix);
            }

            return nonWordChar + suffix;
        }
    }
}