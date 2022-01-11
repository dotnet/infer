// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using System.Reflection;
using System.Diagnostics;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Factors.Attributes;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Compiler.Transforms;
using Microsoft.ML.Probabilistic.Compiler.Reflection;

namespace Microsoft.ML.Probabilistic.Compiler.Visualizers
{
    /// <summary>
    /// Provides a view of the factors available to Infer.NET and their level of support
    /// for different inference algorithms.
    /// </summary>
    internal class DefaultFactorManager
    {
        private readonly FactorManager factorManager = new FactorManager();

        public IAlgorithm[] algs = { new VariationalMessagePassing(), new ExpectationPropagation() };

        /// <summary>
        /// If true, evidence methods are required in order to have "Full Support".
        /// </summary>
        public bool ShowMissingEvidences;

        public DefaultFactorManager(IAlgorithm[] algs)
        {
            this.algs = algs;
        }

        public void Show()
        {
            var generatedHtml = GenerateNewFactorTable();
            var filename = SaveHtml(generatedHtml);
            OpenFactorTable(filename);
        }

        private string GenerateNewFactorTable()
        {
            List<Tuple<string, FactorManager.FactorInfo>> factorList = new List<Tuple<string, FactorManager.FactorInfo>>();
            IEnumerable<FactorManager.FactorInfo> factorInfos = FactorManager.GetFactorInfos();

            foreach (FactorManager.FactorInfo info in factorInfos)
            {
                MethodInfo method = info.Method;
                // omit obsolete, unsupported, and hidden factors
                if (HiddenAttribute.IsDefined(method) ||
                    HiddenAttribute.IsDefined(method.DeclaringType) ||
                    Attribute.IsDefined(method, typeof(ObsoleteAttribute))) continue;
                string itemName = StringUtil.MethodFullNameToString(method);
                ////if (itemName != "EnumSupport.AreEqual<TEnum>") continue;
                factorList.Add(new Tuple<string, FactorManager.FactorInfo>(itemName, info));
            }
            factorList.Sort((elem1, elem2) => elem1.Item2.ToString().CompareTo(elem2.Item2.ToString()));

            string indent = "            ";
            string separator = Environment.NewLine + indent;
            string html = $@"<!DOCTYPE html>
<html>
    <head>
    {AddStyles(this.algs.Length + 2)}
    </head>
    <body>
        <div lang=""en"" class=""container"">
            <div class=""box-header"">Factor</div>
            <div class=""box-header"">Arguments</div>  
            {AddAlgorithmsHeaders()}
            {string.Join(separator, factorList.Select(i => GetFactorHtml(i.Item1, i.Item2)))}
        </div>
    </body>
</html>";
            return html;
        }

        private string AddAlgorithmsHeaders()
        {
            var headers = new StringBuilder();
            foreach (IAlgorithm alg in this.algs) headers.Append($@"<div class=""box-header"">{alg.Name}</div>");
            return headers.ToString();
        }

        private string GetFactorHtml(string name, FactorManager.FactorInfo info)
        {
            Console.WriteLine($"Scanning {info}");
            var currentFactor = new StringBuilder();
            MethodInfo method = info.Method;
            //if (method.Name != "Logistic") continue;

            StringBuilder args = new StringBuilder();
            foreach (ParameterInfo pi in method.GetParameters())
            {
                if (args.Length > 0) args.Append(",</span> ");
                args.Append($"<span>{StringUtil.EscapeXmlCharacters(StringUtil.TypeToString(pi.ParameterType)) + " " + pi.Name}");
            }
            args.Append("</span>");

            string boldString = info.IsDeterministicFactor ? "" : " text-bold";
            currentFactor.Append($@"<div class=""box-left{boldString}"">{StringUtil.EscapeXmlCharacters(name)}</div><div class=""box-left"">{args}</div>");

            foreach (IAlgorithm alg in this.algs)
            {
                ICollection<StochasticityPattern> patterns =
                    GetAlgorithmPatterns(alg, info, ShowMissingEvidences, out QualityBand minQB, out QualityBand modeQB, out QualityBand maxQB);
                if (patterns.Count == 0)
                {
                    // not implemented
                    currentFactor.Append($@"<div class=""box""></div>");
                    continue;
                }
                // Use maximum quality band for the background colour
                QualityBand qb = maxQB;

                var algorithmDiv = new StringBuilder();
                algorithmDiv.Append($@"<div class=""box content-{(
                            qb == QualityBand.Mature
                                ? "mature"
                                : qb == QualityBand.Stable
                                    ? "stable"
                                    : qb == QualityBand.Preview ? "preview" : "experimental")}");

                int partialCount = 0;
                int notSupportedCount = 0;
                StringBuilder sb = new StringBuilder();
                int evidenceCount = 0;
                foreach (StochasticityPattern sp in patterns)
                {
                    if (sp.notSupported != null)
                    {
                        notSupportedCount++;
                        continue;
                    }
                    if (sp.Partial) partialCount++;
                    if (sp.evidenceFound) evidenceCount++;
                    if (sb.Length > 0) sb.Append(",</span> ");
                    sb.Append($"<span>{sp}");
                }
                sb.Append("</span>");
                if (notSupportedCount > 0)
                {
                    sb.Append(" <span> Cannot support:</span>");
                    int ct = 0;
                    foreach (StochasticityPattern sp in patterns)
                    {
                        if (sp.notSupported == null) continue;
                        if (ct > 0) sb.Append(",</span> ");
                        sb.Append($"<span>{sp}");
                        ct++;
                    }
                    sb.Append("</span>");
                }

                string algorithmText = sb.ToString();
                int argCount = info.Method.GetParameters().Length;
                if (info.Method.ReturnType != typeof(void)) argCount++;
                int maxPatterns = (int)System.Math.Pow(2, argCount);
                if (info.IsDeterministicFactor) maxPatterns--;  // (D D) -> S doesn't exist for a deterministic factor

                if ((partialCount == 0) && (patterns.Count >= maxPatterns) && (notSupportedCount == 0))
                {
                    algorithmText = "<span>Full support</span>";
                    if (notSupportedCount > 0) algorithmText = " <span>As supported as possible</span>";
                    if (evidenceCount != patterns.Count) algorithmText += "*";
                    algorithmDiv.Append($@" text-green text-bold"">{algorithmText}</div>");
                    currentFactor.Append(algorithmDiv.ToString());
                    continue;
                }
                else if ((notSupportedCount == patterns.Count) && (notSupportedCount > 0))
                {
                    algorithmText = "<span>Cannot support</span>";
                    algorithmDiv.Append($@" text-bold"">{algorithmText}</div>");
                    currentFactor.Append(algorithmDiv.ToString());
                    continue;
                }
                else if (partialCount > 0)
                {
                    algorithmDiv.Append($@" text-red text-bold"">{algorithmText}</div>");
                    currentFactor.Append(algorithmDiv.ToString());
                }
                else
                {
                    algorithmDiv.Append($@" {((notSupportedCount > 0) ? "" : "text-green")}"">{algorithmText}</div>");
                    currentFactor.Append(algorithmDiv.ToString());
                }
            }

            return currentFactor.ToString();
        }

        private string AddStyles(int columnsNumber)
        {
            var gridTemplate = new StringBuilder();
            for (var i = 0; i < columnsNumber; i++)
            {
                gridTemplate.Append("auto ");
            }
            string styles = $@" 
<style>
    * {{
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}
    body {{
      font-size: 1em;
    }}
    span {{
      display: inline-block;
    }}
    .container {{
      display: grid;
      width: 100%;
      grid-template-columns: {gridTemplate};
      max-width: 1700px;
      box-sizing: border-box;
      border-left: 1px solid #aaa;
      border-top: 1px solid #aaa;
    }}
    .box-header {{
      text-overflow:ellipsis;
      overflow:hidden;
      border-bottom: 1px solid #aaa;
      border-right: 1px solid #aaa;
      font-size: 1em;
      font-weight: bold;
      background-color: #ccc;
      padding-left:5px;
      padding-right:5px;
    }}
    .box-left {{
      border-bottom: 1px solid #aaa;
      border-right: 1px solid #aaa;
      font-size: 0.8em;
      white-space:normal;
      word-wrap:break-word;
      padding-left:5px;
      padding-right:5px;
    }}
    .box {{
      border-bottom: 1px solid #aaa;
      border-right: 1px solid #aaa;
      font-size: 0.8em;
      padding-left:5px;
      padding-right:5px;
    }}
    .text-bold {{
      font-weight: bold;
    }}
    .content-mature {{
      background-color: limegreen;
    }}
    .content-stable {{
      background-color: rgb(153, 230, 153);
    }}
    .content-preview {{
      background-color: gold;
    }}
    .content-experimental {{
      background-color: tomato;
    }}
    .text-red {{
      color: darkred;
    }}
    .text-green {{
      color: green;
    }}
    @media all and (max-width: 1300px) {{
      body {{
        font-size: 0.8em;
      }}
    }}
    @media all and (max-width: 1025px) {{
      body {{
        font-size: 0.7em;
      }}
    }}
    @media all and (max-width: 895px) {{
      body {{
        font-size: 0.6em;
      }}
    }}
    @media all and (max-width: 770px) {{
      body {{
        font-size: 0.5em;
      }}
    }}
    @media all and (max-width: 645px) {{
      body {{
        font-size: 0.4em;
      }}
    }}
  </style>";
            return styles;
        }

        private string SaveHtml(string htmlCode)
        {
            string filename = $"factorTable.html";
            System.IO.File.WriteAllText(filename, htmlCode);
            return filename;
        }

        private void OpenFactorTable(string filename)
        {
            try
            {
                var p = new Process
                {
                    StartInfo = new ProcessStartInfo(filename)
                    {
                        UseShellExecute = true
                    }
                };
                p.Start();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Couldn't find a browser. {ex.Message}");
            }
        }

        /// <summary>
        /// Returns a collection of possible type patterns for the fields of info.  Patterns may be incomplete.
        /// </summary>
        /// <param name="alg"></param>
        /// <param name="info"></param>
        /// <param name="ShowMissingEvidences">If true, evidence methods are included in <see cref="StochasticityPattern.neededCount"/> and <see cref="StochasticityPattern.foundCount"/>.</param>
        /// <param name="minQB">The minimum quality band attached to an operator for this factor</param>
        /// <param name="modeQB">The most common quality band attached to operators for this factor</param>
        /// <param name="maxQB">The maximum quality band attached to operators for this factor</param>
        /// <returns></returns>
        private ICollection<StochasticityPattern> GetAlgorithmPatterns(IAlgorithm alg, FactorManager.FactorInfo info, bool ShowMissingEvidences,
                                                                       out QualityBand minQB, out QualityBand modeQB, out QualityBand maxQB)
        {
            bool verbose = false;
            ICollection<StochasticityPattern> patterns = new Set<StochasticityPattern>();
            string suffix = alg.GetOperatorMethodSuffix(new List<ICompilerAttribute>());
            string evidenceMethodName = alg.GetEvidenceMethodName(new List<ICompilerAttribute>());
            IEnumerable<MessageFcnInfo> mfis = null;
            try
            {
                //Console.WriteLine("***************************"+info.Method);
                mfis = info.GetMessageFcnInfos(suffix, null, null);
                try
                {
                    IEnumerable<MessageFcnInfo> evmfis = info.GetMessageFcnInfos(evidenceMethodName, null, null);
                    mfis = mfis.Concat(evmfis);
                }
                catch (MissingMethodException)
                {
                }
            }
            catch (MissingMethodException)
            {
            }
            catch (Exception ex)
            {
                Console.WriteLine("Failed to get mfis for " + info + " " + ex);
            }
            if (mfis == null)
            {
                minQB = QualityBand.Unknown;
                modeQB = QualityBand.Unknown;
                maxQB = QualityBand.Unknown;
                return patterns;
            }
            minQB = QualityBand.Mature;
            maxQB = QualityBand.Unknown;
            Dictionary<QualityBand, int> qbCounts = new Dictionary<QualityBand, int>();
            // Algorithm: Enumerate all message functions to discover the possible distribution types
            // for stochastic arguments.
            foreach (MessageFcnInfo mfi in mfis)
            {
                if (verbose) Trace.WriteLine(mfi.Method);
                StochasticityPattern sp = new StochasticityPattern(info);
                sp.notSupported = mfi.NotSupportedMessage;
                Dictionary<string, Type> parameterTypes = new Dictionary<string, Type>();
                foreach (KeyValuePair<string, Type> kvp in mfi.GetParameterTypes()) parameterTypes[kvp.Key] = kvp.Value;
                string target = mfi.TargetParameter;
                if (!parameterTypes.ContainsKey(target))
                {
                    if (!mfi.PassResultIndex)
                    {
                        parameterTypes[target] = mfi.Method.ReturnType;
                    }
                    else
                    {
                        parameterTypes[target] = mfi.Method.ReturnType.MakeArrayType();
                    }
                }
                else if (alg is GibbsSampling)
                {
                    sp.notSupported = "non-conjugate";
                }
                QualityBand qb = Quality.GetQualityBand(mfi.Method);
                if (qb < minQB) minQB = qb;
                if (qb > maxQB) maxQB = qb;
                if (!qbCounts.ContainsKey(qb)) qbCounts[qb] = 1;
                else qbCounts[qb] = qbCounts[qb] + 1;

                if (info.Method.IsGenericMethodDefinition)
                {
                    // fill in the factor's type parameters from the type arguments of the operator class.
                    IDictionary<string, Type> typeArgs = FactorManager.FactorInfo.GetTypeArguments(mfi.Method.DeclaringType);
                    try
                    {
                        MethodInfo newMethod = FactorManager.FactorInfo.MakeGenericMethod(info.Method, typeArgs);
                        sp.info = FactorManager.GetFactorInfo(newMethod);
                    }
                    catch (Exception)
                    {
                        sp.info = FactorManager.GetFactorInfo(info.Method);
                        if (verbose)
                            Trace.WriteLine("Could not infer generic type parameters of " + StringUtil.MethodFullNameToString(info.Method));
                    }
                    // from now on, sp.info != info
                }

                foreach (string field in sp.info.ParameterNames)
                {
                    ArgInfo ai = new ArgInfo();
                    if (!parameterTypes.ContainsKey(field))
                    {
                        if (verbose) Trace.WriteLine("not found: " + field + " in " + mfi.Method);
                        continue;
                    }
                    ai.factorType = sp.info.ParameterTypes[field];
                    ai.opType = parameterTypes[field];
                    sp.argInfos[field] = ai;
                }

                if (verbose) Trace.WriteLine(sp);
                if (!sp.IsValid())
                {
                    if (verbose) Trace.WriteLine("Invalid stochasticity pattern");
                    continue;
                }
                List<StochasticityPattern> toRemove = new List<StochasticityPattern>();
                foreach (StochasticityPattern sp2 in patterns)
                {
                    if (sp2.IsMoreSpecificThan(sp))
                    {
                        // merge sp2 into sp
                        StochasticityPattern sp3 = sp.Intersect(sp2);
                        if (sp3 == null)
                        {
                            throw new Exception("intersection is null");
                        }
                        sp = sp3;
                        toRemove.Add(sp2);
                    }
                }
                foreach (StochasticityPattern sp2 in toRemove)
                {
                    patterns.Remove(sp2);
                }
                patterns.Add(sp);
            }
            int qbCnt = -1;
            modeQB = QualityBand.Unknown;
            foreach (KeyValuePair<QualityBand, int> kvp in qbCounts)
            {
                if (kvp.Value > qbCnt)
                {
                    qbCnt = kvp.Value;
                    modeQB = kvp.Key;
                }
            }

            modeQB = minQB;
            patterns = IntersectPatterns(patterns);
            patterns = GetCompletePatterns(patterns, suffix);
            patterns = AddDeterministicPatterns(patterns);
            VerifyPatterns(patterns, suffix, evidenceMethodName, ShowMissingEvidences);
            return RemoveDuplicatePatterns(patterns);
            //return GetBestPatternPlusDeterministic(patterns);
        }

        private static List<StochasticityPattern> RemoveDuplicatePatterns(IEnumerable<StochasticityPattern> patterns)
        {
            var result = new List<StochasticityPattern>();
            foreach (var pattern in patterns.OrderByDescending(sp => sp.foundCount))
            {
                if (!result.Any(sp => sp.IsSamePattern(pattern)))
                {
                    result.Add(pattern);
                }
            }
            return result;
        }

        /// <summary>
        /// Create a new set of patterns containing the closure of all pairwise intersections of patterns.
        /// </summary>
        /// <param name="patterns"></param>
        /// <returns></returns>
        private static Set<StochasticityPattern> IntersectPatterns(IEnumerable<StochasticityPattern> patterns)
        {
            Set<StochasticityPattern> result = new Set<StochasticityPattern>();
            result.AddRange(patterns);
            bool changed;
            do
            {
                int count = result.Count;
                AddIntersections(result);
                changed = (result.Count != count);
                break;
            } while (changed);
            return result;
        }

        /// <summary>
        /// Modify the collection by adding all pairwise intersections of patterns.
        /// </summary>
        /// <param name="patterns"></param>
        private static void AddIntersections(ICollection<StochasticityPattern> patterns)
        {
            List<StochasticityPattern> origPatterns = new List<StochasticityPattern>(patterns);
            foreach (StochasticityPattern sp in origPatterns)
            {
                foreach (StochasticityPattern sp2 in origPatterns)
                {
                    if (sp != sp2)
                    {
                        StochasticityPattern sp3 = sp.Intersect(sp2);
                        if (sp3 != null) patterns.Add(sp3);
                    }
                }
            }
        }

        private static StochasticityPattern GetStochasticityPattern(FactorManager.FactorInfo info, MessageFcnInfo mfi)
        {
            StochasticityPattern sp = new StochasticityPattern(info);
            sp.notSupported = mfi.NotSupportedMessage;
            Dictionary<string, Type> parameterTypes = new Dictionary<string, Type>();
            foreach (KeyValuePair<string, Type> kvp in mfi.GetParameterTypes()) parameterTypes[kvp.Key] = kvp.Value;
            string target = mfi.TargetParameter;
            if (!parameterTypes.ContainsKey(target))
            {
                if (!mfi.PassResultIndex)
                {
                    parameterTypes[target] = mfi.Method.ReturnType;
                }
                else
                {
                    parameterTypes[target] = mfi.Method.ReturnType.MakeArrayType();
                }
            }

            if (info.Method.IsGenericMethodDefinition)
            {
                // fill in the factor's type parameters from the type arguments of the operator class.
                IDictionary<string, Type> typeArgs = FactorManager.FactorInfo.GetTypeArguments(mfi.Method.DeclaringType);
                try
                {
                    MethodInfo newMethod = FactorManager.FactorInfo.MakeGenericMethod(info.Method, typeArgs);
                    sp.info = FactorManager.GetFactorInfo(newMethod);
                }
                catch (Exception)
                {
                    sp.info = FactorManager.GetFactorInfo(info.Method);
                }
                // from now on, sp.info != info
            }

            foreach (string field in sp.info.ParameterNames)
            {
                ArgInfo ai = new ArgInfo();
                if (!parameterTypes.ContainsKey(field))
                {
                    //Console.WriteLine("not found: " + field + " in " + mfi.Method);
                    continue;
                }
                ai.factorType = sp.info.ParameterTypes[field];
                ai.opType = parameterTypes[field];
                sp.argInfos[field] = ai;
            }
            return sp;
        }

        private static ICollection<StochasticityPattern> GetBestPatternPlusDeterministic(ICollection<StochasticityPattern> patterns)
        {
            StochasticityPattern bestp = GetBestPattern(patterns);
            if (bestp == null) return patterns;
            var result = new Set<StochasticityPattern>();
            result.Add(bestp);
            result.AddRange(bestp.deterministicPatterns);
            return result;
        }

        // Returns null if patterns is empty
        private static StochasticityPattern GetBestPattern(IEnumerable<StochasticityPattern> patterns)
        {
            StochasticityPattern bestp = null;
            int best = -1;
            foreach (StochasticityPattern sp in patterns)
            {
                int count = 0;
                if (!sp.Partial) count += 10;
                foreach (StochasticityPattern sp2 in sp.deterministicPatterns)
                {
                    if (!sp2.Partial) count++;
                }
                if (count > best)
                {
                    best = count;
                    bestp = sp;
                }
            }
            return bestp;
        }

        private static Set<StochasticityPattern> GetCompletePatterns(IEnumerable<StochasticityPattern> patterns, string suffix)
        {
            Set<StochasticityPattern> pendingPatterns;
            Set<StochasticityPattern> completePatterns = new Set<StochasticityPattern>();
            do
            {
                pendingPatterns = new Set<StochasticityPattern>();
                foreach (StochasticityPattern sp in patterns)
                {
                    if (sp.IsComplete) completePatterns.Add(sp);
                    // find a missing field
                    foreach (string field in sp.info.ParameterNames)
                    {
                        pendingPatterns.AddRange(GetCompletePatterns(sp, field, suffix));
                    }
                }
                patterns = pendingPatterns;
            } while (pendingPatterns.Count > 0);
            return completePatterns;
        }

        private static IEnumerable<StochasticityPattern> GetCompletePatterns(StochasticityPattern sp, string field, string suffix)
        {
            Dictionary<string, Type> parameterTypes = sp.GetParameterTypes();
            IEnumerable<MessageFcnInfo> mfis;
            for (int trial = 0; trial < 2; trial++)
            {
                try
                {
                    if (sp.argInfos.ContainsKey(field))
                    {
                        if (trial == 0)
                        {
                            parameterTypes["result"] = parameterTypes[field];
                            parameterTypes.Remove("resultIndex");
                        }
                        else if (trial == 1)
                        {
                            Type elementType = Util.GetElementType(parameterTypes[field]);
                            if (elementType == null) break;
                            parameterTypes["result"] = elementType;
                            parameterTypes["resultIndex"] = typeof(int);
                        }
                    }
                    mfis = sp.info.GetMessageFcnInfos(suffix, field, parameterTypes);
                }
                catch
                {
                    mfis = new List<MessageFcnInfo>();
                }
                foreach (MessageFcnInfo mfi in mfis)
                {
                    StochasticityPattern sp2 = GetStochasticityPattern(sp.info, mfi);
                    sp2.AddArgInfos(sp);
                    if (sp2.argInfos.Count > sp.argInfos.Count) yield return sp2;
                }
            }
        }

        /// <summary>
        /// Expand the list of patterns by substituting each stochastic type with a deterministic type.
        /// </summary>
        /// <param name="patterns"></param>
        /// <returns></returns>
        private static List<StochasticityPattern> AddDeterministicPatterns(IEnumerable<StochasticityPattern> patterns)
        {
            List<StochasticityPattern> result = new List<StochasticityPattern>(patterns);
            bool changed;
            do
            {
                changed = AddSomeDeterministicPatterns(result);
            } while (changed);
            return result;
        }

        private static bool AddSomeDeterministicPatterns(List<StochasticityPattern> patterns)
        {
            bool changed = false;
            List<StochasticityPattern> origPatterns = new List<StochasticityPattern>(patterns);
            foreach (StochasticityPattern sp in origPatterns)
            {
                foreach (KeyValuePair<string, ArgInfo> entry in sp.argInfos)
                {
                    ArgInfo arg = entry.Value;
                    if (!arg.IsStoch) continue;
                    StochasticityPattern sp2 = new StochasticityPattern(sp.info);
                    //sp2.notSupported = sp.notSupported;
                    foreach (KeyValuePair<string, ArgInfo> entry2 in sp.argInfos)
                    {
                        if (entry2.Key == entry.Key)
                        {
                            // replace type with deterministic type.
                            ArgInfo arg2 = new ArgInfo
                            {
                                factorType = arg.factorType,
                                opType = arg.factorType
                            };
                            sp2.argInfos[entry2.Key] = arg2;
                        }
                        else
                        {
                            sp2.argInfos[entry2.Key] = (ArgInfo)entry2.Value.Clone();
                        }
                    }
                    if (!sp2.IsValid()) continue;
                    int index = FindPattern(patterns, sp2);
                    if (index == -1)
                    {
                        patterns.Add(sp2);
                        changed = true;
                    }
                    else
                    {
                        sp2 = patterns[index];
                    }
                    int oldCount = sp.deterministicPatterns.Count;
                    sp.deterministicPatterns.Add(sp2);
                    sp.deterministicPatterns.AddRange(sp2.deterministicPatterns);
                    changed = changed || (sp.deterministicPatterns.Count != oldCount);
                }
            }
            return changed;
        }

        private static int FindPattern(IEnumerable<StochasticityPattern> patterns, StochasticityPattern sp)
        {
            int index = 0;
            foreach (StochasticityPattern sp2 in patterns)
            {
                if (sp2.IsMoreSpecificThan(sp)) return index;
                index++;
            }
            return -1;
        }

        /// <summary>
        /// Fills in fields of each StochasticityPattern
        /// </summary>
        /// <param name="patterns"></param>
        /// <param name="suffix"></param>
        /// <param name="evidenceMethodName">The algorithm-specific suffix for an evidence method.</param>
        /// <param name="ShowMissingEvidences">If true, evidence methods are included in <see cref="StochasticityPattern.neededCount"/> and <see cref="StochasticityPattern.foundCount"/>.</param>
        private void VerifyPatterns(IEnumerable<StochasticityPattern> patterns,
                                    string suffix, string evidenceMethodName, bool ShowMissingEvidences)
        {
            foreach (StochasticityPattern sp in patterns)
            {
                Dictionary<string, Type> parameterTypes = sp.GetParameterTypes();

                // Find operators for stochastic args
                foreach (string field in sp.info.ParameterNames)
                {
                    if (!sp.argInfos.ContainsKey(field)) continue;
                    ArgInfo ai = sp.argInfos[field];
                    if (!ai.IsStoch) continue;
                    if (field == sp.info.ParameterNames[0] && sp.info.IsDeterministicFactor && sp.IsStochasticFactorOnly()) continue;
                    sp.neededCount++;
                    for (int trial = 0; trial < 3; trial++)
                    {
                        if (trial == 0)
                        {
                            parameterTypes["result"] = ai.opType;
                            parameterTypes.Remove("resultIndex");
                        }
                        else if (trial == 1)
                        {
                            Type elementType;
                            try
                            {
                                elementType = Util.GetElementType(ai.opType);
                            }
                            catch (AmbiguousMatchException)
                            {
                                break;
                            }
                            if (elementType == null) break;
                            parameterTypes["result"] = elementType;
                            parameterTypes["resultIndex"] = typeof(int);
                        }
                        else if (trial == 2)
                        {
                            Type elementType;
                            try
                            {
                                elementType = Util.GetElementType(ai.opType);
                            }
                            catch (AmbiguousMatchException)
                            {
                                break;
                            }
                            if (elementType == null) break;
                            Type daType;
                            try
                            {
                                daType = Distributions.Distribution.MakeDistributionArrayType(elementType, 1);
                            }
                            catch (ArgumentException)
                            {
                                elementType = typeof(Distributions.Gaussian);
                                daType = Distributions.Distribution.MakeDistributionArrayType(elementType, 1);
                            }
                            parameterTypes["result"] = daType;
                            parameterTypes.Remove("resultIndex");
                        }
                        try
                        {
                            MessageFcnInfo mfi2 = sp.info.GetMessageFcnInfo(factorManager, suffix, field, parameterTypes);
                            ai.implemented = true;
                            sp.foundCount++;
                            break;
                        }
                        catch (AmbiguousMatchException)
                        {
                            ai.implemented = true;
                            sp.foundCount++;
                            break;
                        }
                        catch (NotSupportedException nse)
                        {
                            sp.notSupported = nse.Message;
                        }
                        catch (Exception)
                        {
                            ai.implemented = false;
                        }
                    }
                }

                // Find evidence operator
                parameterTypes["result"] = typeof(double);
                parameterTypes.Remove("resultIndex");
                if (ShowMissingEvidences) sp.neededCount++;
                try
                {
                    MessageFcnInfo evidenceFcn = sp.info.GetMessageFcnInfo(factorManager, evidenceMethodName, "", parameterTypes);
                    sp.evidenceFound = true;
                    if (ShowMissingEvidences) sp.foundCount++;
                }
                catch (AmbiguousMatchException)
                {
                    sp.evidenceFound = true;
                    if (ShowMissingEvidences) sp.foundCount++;
                }
                catch (NotSupportedException nse)
                {
                    sp.notSupported = nse.Message;
                }
                catch (Exception)
                {
                }
            }
        }

        /// <summary>
        /// Represents a pattern of stochasticity in the arguments and return value
        /// of a function.
        /// </summary>
        internal class StochasticityPattern
        {
            internal FactorManager.FactorInfo info;
            internal Dictionary<string, ArgInfo> argInfos = new Dictionary<string, ArgInfo>();
            public Set<StochasticityPattern> deterministicPatterns = new Set<StochasticityPattern>();

            internal string notSupported;
            internal int foundCount = 0;
            internal int neededCount = 0;
            internal bool evidenceFound = false;
            internal int nonConstantCount = 0;

            public bool IsComplete
            {
                get { return argInfos.Count >= nonConstantCount; }
            }

            public bool Partial
            {
                get { return foundCount < neededCount; }
            }

            public StochasticityPattern(FactorManager.FactorInfo info)
            {
                this.info = info;
                nonConstantCount = 0;
                foreach (ParameterInfo pi in info.Method.GetParameters())
                {
                    if (!ConstantAttribute.IsDefined(pi))
                        nonConstantCount++;
                }
            }

            public bool IsValid()
            {
                return !(info.IsDeterministicFactor && IsStochasticFactorOnly());
            }

            public bool IsAllDeterministic()
            {
                for (int i = 0; i < info.ParameterNames.Count; i++)
                {
                    if (argInfos.ContainsKey(info.ParameterNames[i]) && argInfos[info.ParameterNames[i]].IsStoch) return false;
                }
                return true;
            }

            /// <summary>
            /// Is the pattern (D D D) -> S?
            /// </summary>
            /// <returns></returns>
            public bool IsStochasticFactorOnly()
            {
                if (info.Method.ReturnType.Equals(typeof(void))) return false;
                if (argInfos.ContainsKey(info.ParameterNames[0]) && !argInfos[info.ParameterNames[0]].IsStoch) return false;
                for (int i = 1; i < info.ParameterNames.Count; i++)
                {
                    if (argInfos.ContainsKey(info.ParameterNames[i]) && argInfos[info.ParameterNames[i]].IsStoch) return false;
                }
                return true;
            }

            public override string ToString()
            {
                StringBuilder sb = new StringBuilder("(");
                int start = 1;
                if (info.Method.ReturnType.Equals(typeof(void))) start = 0;
                for (int i = start; i < info.ParameterNames.Count; i++)
                {
                    if (i > start) sb.Append(" ");
                    if (argInfos.ContainsKey(info.ParameterNames[i]))
                        sb.Append(argInfos[info.ParameterNames[i]]);
                    else
                        sb.Append("C"); // Must be constant
                }
                sb.Append(")");
                if (start > 0)
                {
                    if (argInfos.ContainsKey(info.ParameterNames[0]))
                        sb.Append("->" + argInfos[info.ParameterNames[0]]);
                    else
                        sb.Append("->_");
                }
                string s = sb.ToString();
                if (notSupported != null)
                {
                    return s;
                }
                if (foundCount < neededCount) s = "[" + s + ":" + foundCount + "/" + neededCount + "]";
                if (!evidenceFound) s += "*";
                return s;
            }

            public override bool Equals(object o)
            {
                if (!(o is StochasticityPattern sp)) return false;
                if (sp.info != info) return false;
                foreach (string field in info.ParameterNames)
                {
                    if (!sp.argInfos.ContainsKey(field))
                    {
                        if (argInfos.ContainsKey(field)) return false;
                        else continue;
                    }
                    if (!argInfos.ContainsKey(field)) return false;
                    // Must compare types because a pattern involving Gaussians is not equivalent
                    // to one using Gammas.
                    if (argInfos[field].opType != sp.argInfos[field].opType) return false;
                }
                return true;
            }

            public override int GetHashCode()
            {
                int hash = Hash.Start;
                hash = Hash.Combine(hash, info.GetHashCode());
                foreach (string field in info.ParameterNames)
                {
                    if (!argInfos.ContainsKey(field)) hash = Hash.Combine(hash, -1);
                    else hash = Hash.Combine(hash, argInfos[field].opType.GetHashCode());
                }
                return hash;
            }

            public bool IsSamePattern(StochasticityPattern sp)
            {
                if (sp.info != info) return false;
                foreach (string field in info.ParameterNames)
                {
                    if (!argInfos.ContainsKey(field) || !sp.argInfos.ContainsKey(field)) continue;
                    if (argInfos[field].IsStoch != sp.argInfos[field].IsStoch) return false;
                    // Must compare types because a pattern involving Gaussians is not equivalent
                    // to one using Gammas.
                    //if (argInfos[field].opType != sp.argInfos[field].opType) return false;
                }
                return true;
            }

            public bool IsMoreSpecificThan(StochasticityPattern sp)
            {
                // this pattern is more specific than sp if it has every argument that sp has, with a compatible type
                if (sp.info != info) return false;
                foreach (string field in info.ParameterNames)
                {
                    if (!sp.argInfos.ContainsKey(field)) continue;
                    if (!argInfos.ContainsKey(field)) return false;
                    // Must compare types because a pattern involving Gaussians is not equivalent
                    // to one using Gammas.
                    if (!IsAssignableFrom(argInfos[field].opType, sp.argInfos[field].opType)) return false;
                }
                return true;
            }

            public static bool IsAssignableFrom(Type t1, Type t2)
            {
                if (t1.IsAssignableFrom(t2)) return true;
                else if (t1.IsGenericType && t2.IsGenericType && t1.GetGenericTypeDefinition().Equals(t2.GetGenericTypeDefinition()))
                {
                    Type[] typeArgs = t1.GetGenericArguments();
                    Type[] typeArgs2 = t2.GetGenericArguments();
                    for (int i = 0; i < typeArgs.Length; i++)
                    {
                        if (!IsAssignableFrom(typeArgs[i], typeArgs2[i])) return false;
                    }
                    return true;
                }
                else if (t1.IsGenericParameter && t2.IsGenericParameter)
                {
                    bool ignoreTypeConstraints = true;
                    if (ignoreTypeConstraints) return (t1.Name == t2.Name);
                    else
                    {
                        GenericParameterFactory.Constraints constraints1 =
                            GenericParameterFactory.Constraints.FromTypeParameter(t1);
                        GenericParameterFactory.Constraints constraints2 =
                            GenericParameterFactory.Constraints.FromTypeParameter(t2);
                        return IsAssignableFrom(constraints1, constraints2);
                    }
                }
                else return false;
            }

            /// <summary>
            /// True if c2 is more constrained than c1.
            /// </summary>
            /// <param name="c1"></param>
            /// <param name="c2"></param>
            /// <returns></returns>
            public static bool IsAssignableFrom(GenericParameterFactory.Constraints c1, GenericParameterFactory.Constraints c2)
            {
                if (c1.attributes != c2.attributes) return false;
                if (c1.baseTypeConstraint != null)
                {
                    if (c2.baseTypeConstraint != null)
                    {
                        if (!IsAssignableFrom(c1.baseTypeConstraint, c2.baseTypeConstraint)) return false;
                    }
                    else
                    {
                        return false;
                    }
                }
                foreach (Type t in c1.interfaceConstraints)
                {
                    bool found = false;
                    foreach (Type t2 in c2.interfaceConstraints)
                    {
                        if (IsAssignableFrom(t, t2))
                        {
                            found = true;
                            break;
                        }
                    }
                    if (!found) return false;
                }
                return true;
            }

            internal Dictionary<string, Type> GetParameterTypes()
            {
                Dictionary<string, Type> parameterTypes = new Dictionary<string, Type>();
                foreach (KeyValuePair<string, ArgInfo> entry in argInfos)
                {
                    parameterTypes[entry.Key] = entry.Value.opType;
                }
                return parameterTypes;
            }

            internal void AddArgInfos(StochasticityPattern sp)
            {
                foreach (KeyValuePair<string, ArgInfo> entry in sp.argInfos)
                {
                    if (!argInfos.ContainsKey(entry.Key)) argInfos[entry.Key] = (ArgInfo)entry.Value.Clone();
                }
            }

            /// <summary>
            /// Create a new pattern whose opTypes are the intersection of corresponding opTypes
            /// </summary>
            /// <param name="that"></param>
            /// <returns></returns>
            internal StochasticityPattern Intersect(StochasticityPattern that)
            {
                if (that.info != info) return null;
                StochasticityPattern result = new StochasticityPattern(info);
                result.notSupported = this.notSupported;
                if (result.notSupported == null) result.notSupported = that.notSupported;
                foreach (string field in info.ParameterNames)
                {
                    ArgInfo arg1;
                    that.argInfos.TryGetValue(field, out ArgInfo arg2);
                    if (!argInfos.TryGetValue(field, out arg1))
                    {
                        if (arg2 == null)
                        {
                            continue;
                        }
                        else
                        {
                            result.argInfos[field] = (ArgInfo)arg2.Clone();
                        }
                    }
                    else if (arg2 == null)
                    {
                        result.argInfos[field] = (ArgInfo)arg1.Clone();
                    }
                    else
                    {
                        Type t = IntersectTypes(arg1.opType, arg2.opType);
                        if (t == null) return null;
                        ArgInfo arg = new ArgInfo();
                        arg.factorType = arg1.factorType;
                        arg.opType = t;
                        result.argInfos[field] = arg;
                    }
                }
                return result;
            }

            internal Dictionary<Set<Type>, Type> intersectionCache = new Dictionary<Set<Type>, Type>();

            /// <summary>
            /// Get a type which is the most specific of the input types.
            /// </summary>
            /// <param name="t1"></param>
            /// <param name="t2"></param>
            /// <returns></returns>
            internal Type IntersectTypes(Type t1, Type t2)
            {
                Set<Type> types = new Set<Type>
                {
                    t1,
                    t2
                };
                Type t;
                if (!intersectionCache.TryGetValue(types, out t))
                {
                    t = Binding.IntersectTypes(t1, t2);
                    if (t != null) intersectionCache[types] = t;
                }
                return t;
            }
        }

        internal class ArgInfo : ICloneable
        {
            internal Type factorType; // factor type
            internal Type opType; // operator type
            internal bool implemented = true;

            public bool IsStoch
            {
                get { return !StochasticityPattern.IsAssignableFrom(opType, factorType); }
            }

            public override string ToString()
            {
                //return StringUtil.TypeToString(opType, true, null);
                string s = "";
                //if (!implemented) s = "X";
                if (IsStoch) s += "S";
                else s += "D";
                //s += "[" + factorType.Name + ":" + opType.Name + "]";
                return s;
            }

            public object Clone()
            {
                ArgInfo result = new ArgInfo();
                result.factorType = factorType;
                result.opType = opType;
                result.implemented = implemented;
                return result;
            }
        }
    }
}