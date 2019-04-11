// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Globalization;
    using System.Linq;
    using System.Text;

    using Distributions;
    using Distributions.Automata;
    using Attributes;
    using Utilities;
    using Microsoft.ML.Probabilistic.Collections;
    using System.Collections.Concurrent;

    /// <summary>
    /// A base class for implementations of message passing operations for various forms of the string formatting factor.
    /// </summary>
    /// <typeparam name="TThis">The type of actual class implementing the operation.</typeparam>
    /// <remarks>We need this class to be generic so that for different <typeparamref name="TThis"/> we have different sets of fields.</remarks>
    [Quality(QualityBand.Experimental)]
    [Buffers("toStrReverseMessage")]
    public class StringFormatOpBase<TThis>
        where TThis : StringFormatOpBase<TThis>, new()
    {
        #region Fields and constants

        /// <summary>
        /// Maps a string representing a set of arguments to an automaton that validates a format string with that set of arguments.
        /// </summary>
        private static readonly ConcurrentDictionary<string, StringAutomaton> ArgsToValidatingAutomaton =
            new ConcurrentDictionary<string, StringAutomaton>();

        /// <summary>
        /// An automaton that allows only for non-empty strings. Used to validate arguments.
        /// </summary>
        private static readonly StringAutomaton DisallowEmptyArgsAutomaton =
            StringAutomaton.Repeat(StringAutomaton.ConstantOnElement(1.0, DiscreteChar.Any()));
        
        /// <summary>
        /// An automaton that accepts any string without '{' and '}' in it.
        /// </summary>
        private static readonly StringAutomaton DisallowBracesAutomaton;

        /// <summary>
        /// A transducer that copies any string without '{' and '}' in it.
        /// </summary>
        private static readonly StringTransducer DisallowBracesTransducer;

        #endregion

        #region Constructor

        /// <summary>
        /// Initializes static members of the <see cref="StringFormatOpBase{TThis}"/> class.
        /// </summary>
        static StringFormatOpBase()
        {
            // More general behavior by default
            RequirePlaceholderForEveryArgument = false;

            DiscreteChar noBraces = DiscreteChar.OneOf('{', '}').Complement();
            DisallowBracesAutomaton = StringAutomaton.Constant(1.0, noBraces);
            DisallowBracesTransducer = StringTransducer.Copy(noBraces);

            // Make sure that the static constructor of TThis has been invoked so that TThis sets everything up
            new TThis();
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets or sets a value indicating whether a format string must have all arguments present.
        /// </summary>
        /// <remarks>
        /// Derived classes are expected to initialize the value of this property in their static constructors.
        /// </remarks>
        protected static bool RequirePlaceholderForEveryArgument { get; set; }

        #endregion

        #region EP messages

        public static StringDistribution StrAverageConditional(StringDistribution format, IList<string> args, IList<string> argNames)
        {
            Argument.CheckIfNotNull(args, "args");

            return StrAverageConditional(format, ToAutomatonArray(args), argNames);
        }

        public static StringDistribution StrAverageConditional(StringDistribution format, IList<StringDistribution> args, IList<string> argNames)
        {
            Argument.CheckIfNotNull(format, "format");
            ValidateArguments(args, argNames);

            var allowedArgs = args.Select(arg => arg.GetWorkspaceOrPoint()).ToList();
            return StrAverageConditionalImpl(format, allowedArgs, argNames, withGroups: false, noValidation: false);
        }

        public static StringDistribution StrAverageConditional_NoValidation(
            StringDistribution format, IList<StringDistribution> args, IList<string> argNames)
        {
            Argument.CheckIfNotNull(format, "format");
            ValidateArguments(args, argNames);

            var allowedArgs = args.Select(arg => arg.GetWorkspaceOrPoint()).ToList();
            return StrAverageConditionalImpl(format, allowedArgs, argNames, withGroups: false, noValidation: true);
        }

        public static StringDistribution FormatAverageConditional(StringDistribution str, IList<string> args, IList<string> argNames)
        {
            Argument.CheckIfNotNull(args, "args");

            return FormatAverageConditional(str, ToAutomatonArray(args), argNames);
        }

        public static StringDistribution FormatAverageConditional(StringDistribution str, IList<StringDistribution> args, IList<string> argNames)
        {
            Argument.CheckIfNotNull(str, "str");
            ValidateArguments(args, argNames);

            var allowedArgs = args.Select(arg => arg.GetWorkspaceOrPoint()).ToList();

            // Try optimizations for special cases
            StringDistribution resultDist;
            if (TryOptimizedFormatAverageConditionalImpl(str, allowedArgs, argNames, out resultDist))
            {
                return resultDist;
            }

            // Reverse the process defined by StrAverageConditional
            var placeholderReplacer = GetPlaceholderReplacingTransducer(allowedArgs, argNames, true, false);
            StringAutomaton format = str.IsPointMass
                ? placeholderReplacer.ProjectSource(str.Point)
                : placeholderReplacer.ProjectSource(str.GetWorkspaceOrPoint());
            StringAutomaton validatedFormat = GetValidatedFormatString(format, argNames);
            return StringDistribution.FromWorkspace(validatedFormat);
        }

        public static StringDistribution ToStrReverseMessage(StringDistribution format, IList<StringDistribution> args, IList<string> argNames)
        {
            return ComputeToStrReverseMessage(format, args, argNames, true);
        }

        public static StringDistribution ComputeToStrReverseMessage(StringDistribution format, IList<StringDistribution> args, IList<string> argNames, bool makeEpsilonFree)
        {
            var allowedArgs = args.Select(arg => arg.GetWorkspaceOrPoint()).ToList();
            StringDistribution toStr = StrAverageConditionalImpl(format, allowedArgs, argNames, withGroups: true, noValidation: false);
            if (makeEpsilonFree)
            {
                toStr.GetWorkspaceOrPoint().MakeEpsilonFree();
            }
            return toStr;
        }

        public static TStringDistributionList ArgsAverageConditional<TStringDistributionList>(
            StringDistribution str, StringDistribution format, IList<StringDistribution> args, IList<string> argNames, TStringDistributionList result)
            where TStringDistributionList : class, IList<StringDistribution>
        {
            Argument.CheckIfNotNull(str, "str");
            Argument.CheckIfNotNull(format, "format");
            ValidateArguments(args, argNames);

            StringDistribution toStr = ToStrReverseMessage(format, args, argNames);
            return ArgsAverageConditionalBuffered(str,toStr,args,result);
        }

        private static readonly StringDistribution Uniform = StringDistribution.Uniform();

        public static TStringDistributionList ArgsAverageConditionalBuffered<TStringDistributionList>(
            StringDistribution str, StringDistribution toStrReverseMessage, IList<StringDistribution> args, TStringDistributionList result)
            where TStringDistributionList : class, IList<StringDistribution>
        {
            ArgsAverageConditionalBufferedWithNulls(str, toStrReverseMessage, args, result);
            for(var i=0; i< result.Count; i++)
                if (result[i] == null)
                    result[i] = Uniform;
            return result;
        }

        public static TStringDistributionList ArgsAverageConditionalBufferedWithNulls<TStringDistributionList>(
            StringDistribution str, StringDistribution toStrReverseMessage, IList<StringDistribution> args, TStringDistributionList result)
            where TStringDistributionList : class, IList<StringDistribution>
        {
            var toStrTimesStr = toStrReverseMessage.Product(str);
            var groups = toStrTimesStr.GetGroups();
            bool isZero = toStrTimesStr.IsZero();

            for (var i = 0; i < args.Count; ++i)
            {
                // Compute forward message assuming nothing about the argument
                if (isZero)
                {
                    result[i] = toStrTimesStr; // i.e. zero, but cheaper than creating a fresh Zero()
                }
                else
                {
                    var argumentGroup = i + 1;
                    StringDistribution group;
                    if (!groups.TryGetValue(argumentGroup, out group))
                    {
                        // The argument is not mentioned in the product distribution, so we know nothing about it
                        // JW: I removed the constraint that non-mentioned args must be non-empty, since it doesn't achieve anything useful.
                        group = null;
                    }
                    else
                    {
                        group.GetWorkspaceOrPoint().TryNormalizeValues();
                    }
                    result[i] = group;
                }
            }

            return result;
        }

        [Skip]
        public static double LogEvidenceRatio(StringDistribution str)
        {
            return 0;
        }

        public static double LogEvidenceRatio(string str, StringDistribution format, IList<StringDistribution> args, IList<string> argNames)
        {
            Argument.CheckIfNotNull(str, "str");

            StringDistribution toStr = StrAverageConditional(format, args, argNames);
            return toStr.GetLogProb(str);
        }

        public static double LogEvidenceRatio(string str, StringDistribution format, IList<string> args, IList<string> argNames)
        {
            Argument.CheckIfNotNull(args, "args");

            return LogEvidenceRatio(str, format, ToAutomatonArray(args), argNames);
        }

        #endregion

        #region Helpers

        /// <summary>
        /// An implementation of <see cref="FormatAverageConditional(StringDistribution, IList{StringDistribution}, IList{string})"/>
        /// specialized for some cases for performance reasons.
        /// </summary>
        /// <param name="str">The message from <c>str</c>.</param>
        /// <param name="allowedArgs">The message from <c>args</c>, truncated to allowed values and converted to automata.</param>
        /// <param name="argNames">The names of the arguments.</param>
        /// <param name="resultDist">The computed result.</param>
        /// <returns>
        /// <see langword="true"/> if there is an optimized implementation available for the provided parameters,
        /// and <paramref name="resultDist"/> has been computed using it.
        /// <see langword="false"/> otherwise.
        /// </returns>
        /// <remarks>
        /// Supports the case of point mass <paramref name="str"/> and <paramref name="allowedArgs"/>,
        /// where each of the arguments is present in <paramref name="str"/> at most once and the occurrences
        /// are non-overlapping.
        /// </remarks>
        private static bool TryOptimizedFormatAverageConditionalImpl(
            StringDistribution str, IList<StringAutomaton> allowedArgs, IList<string> argNames, out StringDistribution resultDist)
        {
            resultDist = null;

            string[] allowedArgPoints = Util.ArrayInit(allowedArgs.Count, i => allowedArgs[i].TryComputePoint());
            if (!str.IsPointMass || !allowedArgPoints.All(argPoint => argPoint != null && SubstringOccurrencesCount(str.Point, argPoint) <= 1))
            {
                // Fall back to the general case
                return false;
            }

            // Obtain arguments present in 'str' (ordered by position)
            var argPositions =
               allowedArgPoints.Select((arg, argIndex) => Tuple.Create(argIndex, str.Point.IndexOf(arg, StringComparison.Ordinal)))
                   .Where(t => t.Item2 != -1)
                   .OrderBy(t => t.Item2)
                   .ToList();

            if (RequirePlaceholderForEveryArgument && argPositions.Count != allowedArgs.Count)
            {
                // Some argument is not in 'str'
                resultDist = StringDistribution.Zero();
                return true;
            }

            StringAutomaton result = StringAutomaton.ConstantOn(1.0, string.Empty);
            int curArgumentIndex = -1;
            int curArgumentPos = -1;
            int curArgumentLength = 1;
            for (int i = 0; i < argPositions.Count; ++i)
            {
                int prevArgumentIndex = curArgumentIndex;
                int prevArgumentPos = curArgumentPos;
                int prevArgumentLength = curArgumentLength;
                curArgumentIndex = argPositions[i].Item1;
                curArgumentPos = argPositions[i].Item2;
                curArgumentLength = allowedArgPoints[curArgumentIndex].Length;

                if (prevArgumentIndex != -1 && curArgumentPos < prevArgumentPos + prevArgumentLength)
                {
                    // It's easier to fall back to the general case in case of overlapping arguments
                    return false;
                }

                // Append the contents of 'str' preceeding the current argument
                result.AppendInPlace(str.Point.Substring(prevArgumentPos + prevArgumentLength, curArgumentPos - prevArgumentPos - prevArgumentLength));

                // The format may have included either the text ot the placeholder
                string argName = "{" + argNames[curArgumentIndex] + "}";
                if (RequirePlaceholderForEveryArgument)
                {
                    result.AppendInPlace(StringAutomaton.ConstantOn(1.0, argName));
                }
                else
                {
                    result.AppendInPlace(StringAutomaton.ConstantOn(1.0, argName, allowedArgPoints[curArgumentIndex]));
                }   
            }

            // Append the rest of 'str'
            result.AppendInPlace(str.Point.Substring(curArgumentPos + curArgumentLength, str.Point.Length - curArgumentPos - curArgumentLength));

            resultDist = StringDistribution.FromWorkspace(result);
            return true;
        }

        /// <summary>
        /// The implementation of <see cref="StrAverageConditional(StringDistribution, IList{StringDistribution}, IList{string})"/>.
        /// </summary>
        /// <param name="format">The message from <c>format</c>.</param>
        /// <param name="allowedArgs">The message from <c>args</c>, truncated to allowed values and converted to automata.</param>
        /// <param name="argNames">The names of the arguments.</param>
        /// <param name="withGroups">Whether the result should mark different arguments with groups.</param>
        /// <param name="noValidation">Whether incorrect format string values should not be pruned.</param>
        /// <returns>The message to <c>str</c>.</returns>
        private static StringDistribution StrAverageConditionalImpl(
            StringDistribution format, IList<StringAutomaton> allowedArgs, IList<string> argNames, bool withGroups, bool noValidation)
        {
            StringDistribution resultDist = TryOptimizedStrAverageConditionalImpl(format, allowedArgs, argNames, withGroups);
            if (resultDist != null)
            {
                return resultDist;
            }
            
            // Check braces for correctness.
            StringAutomaton validatedFormat = format.GetWorkspaceOrPoint();
            if (!noValidation)
            {
                validatedFormat = GetValidatedFormatString(format.GetWorkspaceOrPoint(), argNames);
            }

            // Now replace placeholders with arguments
            var placeholderReplacer = GetPlaceholderReplacingTransducer(allowedArgs, argNames, false, withGroups);
            StringAutomaton str = placeholderReplacer.ProjectSource(validatedFormat);

            return StringDistribution.FromWorkspace(str);
        }

        /// <summary>
        /// An implementation of <see cref="StrAverageConditional(StringDistribution, IList{StringDistribution}, IList{string})"/>
        /// specialized for some cases for performance reasons.
        /// </summary>
        /// <param name="format">The message from <c>format</c>.</param>
        /// <param name="allowedArgs">The message from <c>args</c>, truncated to allowed values and converted to automata.</param>
        /// <param name="argNames">The names of the arguments.</param>
        /// <param name="withGroups">Whether the result should mark different arguments with groups.</param>
        /// <returns>
        /// Result distribution if there is an optimized implementation available for the provided parameters.
        /// <see langword="null"/> otherwise.
        /// </returns>
        /// <remarks>
        /// Supports the case of point mass <paramref name="format"/>.
        /// </remarks>
        private static StringDistribution TryOptimizedStrAverageConditionalImpl(
            StringDistribution format, IList<StringAutomaton> allowedArgs, IList<string> argNames, bool withGroups)
        {
            if (!format.IsPointMass)
            {
                // Fall back to the general case
                return null;
            }
            
            // Check braces for correctness & replace placeholders with arguments simultaneously
            var result = StringAutomaton.Builder.ConstantOn(Weight.One, string.Empty);
            bool[] argumentSeen = new bool[allowedArgs.Count];
            int openingBraceIndex = format.Point.IndexOf("{", StringComparison.Ordinal), closingBraceIndex = -1;
            while (openingBraceIndex != -1)
            {
                // Add the part of the format before the placeholder
                result.Append(StringAutomaton.ConstantOn(1.0, format.Point.Substring(closingBraceIndex + 1, openingBraceIndex - closingBraceIndex - 1)));
                
                // Find next opening and closing braces
                closingBraceIndex = format.Point.IndexOf("}", openingBraceIndex + 1, StringComparison.Ordinal);
                int nextOpeningBraceIndex = format.Point.IndexOf("{", openingBraceIndex + 1, StringComparison.Ordinal);

                // Opening brace must be followed by a closing brace
                if (closingBraceIndex == -1 || (nextOpeningBraceIndex != -1 && nextOpeningBraceIndex < closingBraceIndex))
                {
                    return StringDistribution.Zero();
                }

                string argumentName = format.Point.Substring(openingBraceIndex + 1, closingBraceIndex - openingBraceIndex - 1);
                int argumentIndex = argNames.IndexOf(argumentName);

                // Unknown or previously seen argument found
                if (argumentIndex == -1 || argumentSeen[argumentIndex])
                {
                    return StringDistribution.Zero();
                }

                // Replace the placeholder by the argument
                result.Append(allowedArgs[argumentIndex], withGroups ? argumentIndex + 1 : 0);

                // Mark the argument as 'seen'
                argumentSeen[argumentIndex] = true;

                openingBraceIndex = nextOpeningBraceIndex;
            }

            // There should be no closing braces after the last opening brace
            if (format.Point.IndexOf('}', closingBraceIndex + 1) != -1)
            {
                return StringDistribution.Zero();
            }

            if (RequirePlaceholderForEveryArgument && argumentSeen.Any(seen => !seen))
            {
                // Some argument wasn't present although it was required
                return StringDistribution.Zero();
            }

            // Append the part of the format after the last placeholder
            result.Append(StringAutomaton.ConstantOn(1.0, format.Point.Substring(closingBraceIndex + 1, format.Point.Length - closingBraceIndex - 1)));

            return StringDistribution.FromWorkspace(result.GetAutomaton());
        }

        /// <summary>
        /// Creates a transducer that replaces argument placeholders with the corresponding arguments.
        /// </summary>
        /// <param name="args">The list of arguments.</param>
        /// <param name="argNames">The list of argument names.</param>
        /// <param name="forBackwardMessage">Specifies whether the created transducer should be transposed so that the backward message can be computed.</param>
        /// <param name="withGroups">Specifies whether filled in arguments should be labeled with different groups.</param>
        /// <returns>The created transducer.</returns>
        private static StringTransducer GetPlaceholderReplacingTransducer(
            IList<StringAutomaton> args, IList<string> argNames, bool forBackwardMessage, bool withGroups)
        {
            var alternatives = new List<StringTransducer>();
            for (int argumentIndex = 0; argumentIndex < args.Count; ++argumentIndex)
            {
                StringTransducer alternative;
                if (!forBackwardMessage)
                {
                    alternative = StringTransducer.Consume(argNames[argumentIndex]);
                    alternative.AppendInPlace(StringTransducer.Produce(args[argumentIndex]), withGroups ? argumentIndex + 1 : 0);
                }
                else
                {
                    // After transposing 'Produce' will become 'Consume',
                    // and starting from 'Consume' makes projection computations more efficeint because it allows to detect dead braches earlier
                    Debug.Assert(!withGroups, "Groups are never needed for backward message.");
                    alternative = StringTransducer.Produce(args[argumentIndex]);
                    alternative.AppendInPlace(StringTransducer.Consume(argNames[argumentIndex]));
                }

                alternatives.Add(alternative);
            }

            StringTransducer result = DisallowBracesTransducer.Clone();
            result.AppendInPlace(StringTransducer.ConsumeElement('{'));
            result.AppendInPlace(StringTransducer.Sum(alternatives));
            result.AppendInPlace(StringTransducer.ConsumeElement('}'));
            result = StringTransducer.Repeat(result, minTimes: 0);
            result.AppendInPlace(DisallowBracesTransducer);

            if (forBackwardMessage)
            {
                result = StringTransducer.Transpose(result);
            }

            return result;
        }

        /// <summary>
        /// Multiplies the given format string automaton onto an automaton representing allowed format strings for a given set of arguments.
        /// </summary>
        /// <param name="format">The automaton representing format strings.</param>
        /// <param name="argNames">The list of format string arguments.</param>
        /// <returns>The multiplication results.</returns>
        private static StringAutomaton GetValidatedFormatString(StringAutomaton format, IList<string> argNames)
        {
            Debug.Assert(argNames.Count > 0, "The code below relies on at least one argument being provided.");
            
            var result = new StringAutomaton();
            for (int i = 0; i < argNames.Count; ++i)
            {
                StringAutomaton validatingAutomaton = GetArgumentValidatingAutomaton(i, argNames);
                result.SetToProduct(i == 0 ? format : result, validatingAutomaton);
                result.ClearGroups();
                result.TrySetToConstantOnSupportOfLog(0.0, result);
            }

            return result;
        }
        
        /// <summary>
        /// Creates an automaton for validating the correctness of argument placeholders for a given argument.
        /// </summary>
        /// <param name="argToValidateIndex">The index of the argument to validate.</param>
        /// <param name="argNames">The names of all arguments.</param>
        /// <returns>The created automaton.</returns>
        private static StringAutomaton GetArgumentValidatingAutomaton(int argToValidateIndex, IList<string> argNames)
        {
            Debug.Assert(
                argNames != null && argNames.All(name => !string.IsNullOrEmpty(name)),
                "A valid array of argument names must be provided.");
            Debug.Assert(
                argToValidateIndex >= 0 && argToValidateIndex < argNames.Count,
                "The provided argument index must be a valid index in the argument name array.");
            
            string argListKey = ArgumentListToDictionaryKey(argToValidateIndex, argNames);
            StringAutomaton result;
            if (ArgsToValidatingAutomaton.TryGetValue(argListKey, out result))
            {
                return result;
            }

            // Accepts placeholder for the current argument
            StringAutomaton checkBracesForCurrentArg = StringAutomaton.ConstantOn(1.0, "{" + argNames[argToValidateIndex] + "}");

            StringAutomaton checkBracesForOtherArgs = DisallowBracesAutomaton.Clone();
            if (argNames.Count > 1)
            {
                // Skips placeholders for every argument except the current one
                StringAutomaton skipOtherArgs = StringAutomaton.ConstantOnElement(1.0, '{');
                skipOtherArgs.AppendInPlace(StringAutomaton.ConstantOn(1.0, argNames.Where((arg, index) => index != argToValidateIndex)));
                skipOtherArgs.AppendInPlace(StringAutomaton.ConstantOnElement(1.0, '}'));

                // Accepts placeholders for arguments other than current, with arbitrary intermediate text
                checkBracesForOtherArgs.AppendInPlace(skipOtherArgs);
                checkBracesForOtherArgs = StringAutomaton.Repeat(checkBracesForOtherArgs, minTimes: 0);
                checkBracesForOtherArgs.AppendInPlace(DisallowBracesAutomaton);
            }

            // Checks the placeholder for the current argument, then skips placeholders for other arguments
            StringAutomaton validateArgumentThenOtherArguments = checkBracesForCurrentArg.Clone();
            validateArgumentThenOtherArguments.AppendInPlace(checkBracesForOtherArgs);
            if (!RequirePlaceholderForEveryArgument)
            {
                // Make this block optional
                validateArgumentThenOtherArguments = StringAutomaton.Sum(
                    validateArgumentThenOtherArguments,
                    StringAutomaton.ConstantOn(1.0, string.Empty));
            }

            // Accepts placeholders for arguments other then current, then for the current argument, then again other placeholders
            result = checkBracesForOtherArgs.Clone();
            result.AppendInPlace(validateArgumentThenOtherArguments);

            result.TryDeterminize();
            ArgsToValidatingAutomaton[argListKey] = result;

            return result;
        }

        /// <summary>
        /// Computes the number of occurrences of a substring in a string.
        /// </summary>
        /// <param name="str">The string.</param>
        /// <param name="substr">The substring.</param>
        /// <returns>The number of occurrences.</returns>
        private static int SubstringOccurrencesCount(string str, string substr)
        {
            int count = 0, prevPos = str.IndexOf(substr, StringComparison.Ordinal);
            while ((prevPos != -1) && (prevPos < str.Length - 1))
            {
                ++count;
                prevPos = str.IndexOf(substr, prevPos + 1, StringComparison.Ordinal);
            }

            return count;
        }

        /// <summary>
        /// Creates a key for <see cref="ArgsToValidatingAutomaton"/>, which is a dictionary
        /// used to cache the values of <see cref="GetArgumentValidatingAutomaton"/>.
        /// </summary>
        /// <param name="argIndexToValidate">The index of the argument to validate.</param>
        /// <param name="argNames">The names of all arguments.</param>
        /// <returns>The key for <see cref="ArgsToValidatingAutomaton"/>.</returns>
        private static string ArgumentListToDictionaryKey(int argIndexToValidate, IList<string> argNames)
        {
            var result = new StringBuilder();
            result.Append(argNames[argIndexToValidate]);
            for (int i = 0; i < argNames.Count; ++i)
            {
                result.Append("{");
                result.Append(argNames[i]);
                result.Append("}");
            }

            return result.ToString();
        }

        /// <summary>
        /// Validates a message from <c>args</c> and the corresponding array of argument names.
        /// </summary>
        /// <param name="args">The message from <c>args</c>.</param>
        /// <param name="argNames">The names of the arguments.</param>
        private static void ValidateArguments(IList<StringDistribution> args, IList<string> argNames)
        {
            Argument.CheckIfNotNull(args, "args");
            Argument.CheckIfNotNull(argNames, "argNames");
            Argument.CheckIfValid(args.Count > 0, "args", "There must be at least one argument provided."); // TODO: relax?
            Argument.CheckIfValid(args.Count == argNames.Count, "The number of arguments and argument names must be the same.");
            Argument.CheckIfValid(argNames.Distinct().Count() == argNames.Count, "argNames", "There must not be two arguments with the same name.");
            Argument.CheckIfValid(argNames.All(arg => arg.IndexOfAny(new[] { '{', '}' }) == -1), "argNames", "Argument names must not contain braces.");
        }

        /// <summary>
        /// Converts a list of string to an array of point mass automata.
        /// </summary>
        /// <param name="strings">The list of strings.</param>
        /// <returns>The created automata array.</returns>
        private static StringDistribution[] ToAutomatonArray(IList<string> strings)
        {
            Debug.Assert(strings != null, "A valid string list must be provided.");

            return Util.ArrayInit(strings.Count, i => StringDistribution.PointMass(strings[i]));
        }

        #endregion
    }
    
    /// <summary>
    /// An implementation of message passing operations for the string formatting factor that supports argument names and
    /// requires every argument placeholder to be present in a format string.
    /// </summary>
    [Quality(QualityBand.Experimental)]
    public class StringFormatOp_RequireEveryPlaceholder : StringFormatOpBase<StringFormatOp_RequireEveryPlaceholder>
    {
        /// <summary>
        /// Initializes static members of the <see cref="StringFormatOp_RequireEveryPlaceholder"/> class.
        /// </summary>
        static StringFormatOp_RequireEveryPlaceholder()
        {
            StringFormatOpBase<StringFormatOp_RequireEveryPlaceholder>.RequirePlaceholderForEveryArgument = true;
        }
    }

    /// <summary>
    /// An implementation of message passing operations for the string formatting factor that supports argument names and
    /// does not require every argument placeholder to be present in a format string.
    /// </summary>
    [Quality(QualityBand.Experimental)]
    public class StringFormatOp_AllowMissingPlaceholders : StringFormatOpBase<StringFormatOp_AllowMissingPlaceholders>
    {
        /// <summary>
        /// Initializes static members of the <see cref="StringFormatOp_AllowMissingPlaceholders"/> class.
        /// </summary>
        static StringFormatOp_AllowMissingPlaceholders()
        {
            StringFormatOpBase<StringFormatOp_AllowMissingPlaceholders>.RequirePlaceholderForEveryArgument = false;
        }
    }

    /// <summary>
    /// A base class for implementations of message passing operations for string formatting factors that don't support specifying argument names.
    /// </summary>
    /// <typeparam name="TStringFormatOp">The type of the underlying implementation that supports argument names.</typeparam>
    [Quality(QualityBand.Experimental)]
    public class StringFormatOpBase_NoArgumentNames<TStringFormatOp>
        where TStringFormatOp : StringFormatOpBase<TStringFormatOp>, new()
    {
        /// <summary>
        /// Maps the number of argument to an array of default argument names ("0", "1" and so on).
        /// </summary>
        private static ReadOnlyArray<string>?[] ArgumentCountToNames = new ReadOnlyArray<string>?[0];

        public static StringDistribution StrAverageConditional(StringDistribution format, IList<string> args)
        {
            Argument.CheckIfNotNull(args, "args");

            return StringFormatOpBase<TStringFormatOp>.StrAverageConditional(format, args, GetArgumentNames(args.Count));
        }

        public static StringDistribution StrAverageConditional(StringDistribution format, IList<StringDistribution> args)
        {
            Argument.CheckIfNotNull(args, "args");

            return StringFormatOpBase<TStringFormatOp>.StrAverageConditional(format, args, GetArgumentNames(args.Count));
        }

        public static StringDistribution FormatAverageConditional(StringDistribution str, IList<string> args)
        {
            Argument.CheckIfNotNull(args, "args");

            return StringFormatOpBase<TStringFormatOp>.FormatAverageConditional(str, args, GetArgumentNames(args.Count));
        }

        public static StringDistribution FormatAverageConditional(StringDistribution str, IList<StringDistribution> args)
        {
            Argument.CheckIfNotNull(args, "args");

            return StringFormatOpBase<TStringFormatOp>.FormatAverageConditional(str, args, GetArgumentNames(args.Count));
        }

        public static TStringDistributionList ArgsAverageConditional<TStringDistributionList>(
            StringDistribution str, StringDistribution format, IList<StringDistribution> args, TStringDistributionList result)
            where TStringDistributionList : class, IList<StringDistribution>
        {
            Argument.CheckIfNotNull(args, "args");

            return StringFormatOpBase<TStringFormatOp>.ArgsAverageConditional(str, format, args, GetArgumentNames(args.Count), result);
        }

        public static StringDistribution ToStrReverseMessage(StringDistribution format, IList<StringDistribution> args)
        {
            Argument.CheckIfNotNull(args, "args");

            return StringFormatOpBase<TStringFormatOp>.ToStrReverseMessage(format, args, GetArgumentNames(args.Count));
        }

        [Skip]
        public static double LogEvidenceRatio(StringDistribution str)
        {
            return 0;
        }

        public static double LogEvidenceRatio(string str, StringDistribution format, IList<StringDistribution> args)
        {
            Argument.CheckIfNotNull(args, "args");

            return StringFormatOpBase<TStringFormatOp>.LogEvidenceRatio(str, format, args, GetArgumentNames(args.Count));
        }

        public static double LogEvidenceRatio(string str, StringDistribution format, IList<string> args)
        {
            Argument.CheckIfNotNull(args, "args");

            return StringFormatOpBase<TStringFormatOp>.LogEvidenceRatio(str, format, args, GetArgumentNames(args.Count));
        }

        /// <summary>
        /// Generates an array of default argument names ("0", "1" and so on) given the number of arguments.
        /// </summary>
        /// <param name="argCount">The number of arguments.</param>
        /// <returns>The generated array of argument names.</returns>
        private static string[] GetArgumentNames(int argCount)
        {
            string[] result = null;
            var cache = GetCache(argCount);
            var readOnlyResult = cache[argCount];
            if (readOnlyResult == null)
            {
                result = Util.ArrayInit(argCount, i => i.ToString(CultureInfo.InvariantCulture));
                ArgumentCountToNames[argCount] = result;
            }
            else
            {
                result = new List<string>(readOnlyResult).ToArray();
            }

            return result;
        }

        private static ReadOnlyArray<string>?[] GetCache(int minSize)
        {
            lock (ArgumentCountToNames)
            {
                if (minSize >= ArgumentCountToNames.Length)
                {
                    var oldCache = ArgumentCountToNames;
                    var newCache = new ReadOnlyArray<string>?[minSize + 1];
                    Array.Copy(ArgumentCountToNames, 0, newCache, 0, ArgumentCountToNames.Count());
                    ArgumentCountToNames = newCache;
                }

                return ArgumentCountToNames;
            }

        }
    }

    /// <summary>
    /// An implementation of message passing operations for the string formatting factor that does not support argument names and
    /// requires every argument placeholder to be present in a format string.
    /// </summary>
    [Quality(QualityBand.Experimental)]
    [FactorMethod(typeof(Factor), "StringFormat", Default = true)]
    public class StringFormatOp_RequireEveryPlaceholder_NoArgumentNames :
        StringFormatOpBase_NoArgumentNames<StringFormatOp_RequireEveryPlaceholder>
    {
    }

    /// <summary>
    /// An implementation of message passing operations for the string formatting factor that does not support argument names and
    /// requires every argument placeholder to be present in a format string.
    /// </summary>
    [Quality(QualityBand.Experimental)]
    [FactorMethod(typeof(Factor), "StringFormat", Default = false)]
    public class StringFormatOp_AllowMissingPlaceholders_NoArgumentNames :
        StringFormatOpBase_NoArgumentNames<StringFormatOp_AllowMissingPlaceholders>
    {
    }
}
