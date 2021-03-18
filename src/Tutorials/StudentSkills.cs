// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Math;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tutorials
{
    [Example("Applications", "Cognitive Assessment Models for Inferring Student Skills")]
    public class StudentSkills
    {
        // Sample data from a DINA/NIDA model and then use Infer.NET to recover the parameters.
        public void Run()
        {
            InferenceEngine engine = new InferenceEngine();
            if (!(engine.Algorithm is Algorithms.ExpectationPropagation))
            {
                Console.WriteLine("This example only runs with Expectation Propagation");
                return;
            }

            bool useDina = true;
            Beta slipPrior = new Beta(1, 10);
            Beta guessPrior = new Beta(1, 10);
            Rand.Restart(0);
            int nStudents = 100;
            int nQuestions = 20;
            int nSkills = 3;
            int[][] skillsRequired = new int[nQuestions][];
            for (int q = 0; q < nQuestions; q++)
            {
                // each question requires a random set of skills
                int[] skills = Rand.Perm(nSkills);
                int n = Rand.Int(nSkills) + 1;
                skillsRequired[q] = Util.ArrayInit(n, i => skills[i]);
                Console.WriteLine("skillsRequired[{0}] = {1}", q, Util.CollectionToString(skillsRequired[q]));
            }

            double[] pSkill, slip, guess;
            bool[][] hasSkill;
            VariableArray<double> slipVar, guessVar, pSkillVar;
            VariableArray<VariableArray<bool>, bool[][]> hasSkillVar;
            if (useDina)
            {
                bool[][] responses = DinaSample(nStudents, nSkills, skillsRequired, slipPrior, guessPrior, out pSkill, out slip, out guess, out hasSkill);
                DinaModel(responses, nSkills, skillsRequired, slipPrior, guessPrior, out pSkillVar, out slipVar, out guessVar, out hasSkillVar);
            }
            else
            {
                bool[][] responses = NidaSample(nStudents, nSkills, skillsRequired, slipPrior, guessPrior, out pSkill, out slip, out guess, out hasSkill);
                NidaModel(responses, nSkills, skillsRequired, slipPrior, guessPrior, out pSkillVar, out slipVar, out guessVar, out hasSkillVar);
            }

            engine.NumberOfIterations = 10;
            Bernoulli[][] hasSkillPost = engine.Infer<Bernoulli[][]>(hasSkillVar);
            int numErrors = 0;
            for (int i = 0; i < nStudents; i++)
            {
                for (int s = 0; s < nSkills; s++)
                {
                    if (hasSkill[i][s] != (hasSkillPost[i][s].LogOdds > 0))
                    {
                        numErrors++;
                    }
                }
            }

            Console.WriteLine("{0:0}% of skills recovered correctly", 100.0 - 100.0 * numErrors / (nStudents * nSkills));
            Beta[] pSkillPost = engine.Infer<Beta[]>(pSkillVar);
            Beta[] slipPost = engine.Infer<Beta[]>(slipVar);
            Beta[] guessPost = engine.Infer<Beta[]>(guessVar);
            for (int s = 0; s < nSkills; s++)
            {
                Console.WriteLine("pSkill[{0}] = {1} (sampled from {2})", s, pSkillPost[s], pSkill[s].ToString("g4"));
            }

            for (int i = 0; i < System.Math.Min(3, slipPost.Length); i++)
            {
                Console.WriteLine("slip[{0}] = {1} (sampled from {2})", i, slipPost[i], slip[i].ToString("g4"));
            }

            for (int i = 0; i < System.Math.Min(3, guessPost.Length); i++)
            {
                Console.WriteLine("guess[{0}] = {1} (sampled from {2})", i, guessPost[i], guess[i].ToString("g4"));
            }
        }

        // Sample data from the DINA model
        public static bool[][] DinaSample(
            int nStudents,
            int nSkills,
            int[][] skillsRequired,
            Beta slipPrior,
            Beta guessPrior,
            out double[] pSkillOut,
            out double[] slip,
            out double[] guess,
            out bool[][] hasSkill)
        {
            int nQuestions = skillsRequired.Length;
            double[] pSkill = Util.ArrayInit(nSkills, q => Rand.Double());
            slip = Util.ArrayInit(nQuestions, q => slipPrior.Sample());
            guess = Util.ArrayInit(nQuestions, q => guessPrior.Sample());
            hasSkill = Util.ArrayInit(nStudents, t => Util.ArrayInit(nSkills, s => Rand.Double() < pSkill[s]));
            bool[][] responses = new bool[nStudents][];
            for (int t = 0; t < nStudents; t++)
            {
                responses[t] = new bool[nQuestions];
                for (int q = 0; q < nQuestions; q++)
                {
                    bool hasAllSkills = Factor.AllTrue(Collection.Subarray(hasSkill[t], skillsRequired[q]));
                    if (hasAllSkills)
                    {
                        responses[t][q] = (Rand.Double() > slip[q]);
                    }
                    else
                    {
                        responses[t][q] = (Rand.Double() < guess[q]);
                    }
                }
            }

            pSkillOut = pSkill;
            return responses;
        }

        // Construct a DINA model in Infer.NET
        public static void DinaModel(
            bool[][] responsesData,
            int nSkills,
            int[][] skillsRequired,
            Beta slipPrior,
            Beta guessPrior,
            out VariableArray<double> pSkill,
            out VariableArray<double> slip,
            out VariableArray<double> guess, 
            out VariableArray<VariableArray<bool>, bool[][]> hasSkill)
        {
            // The Infer.NET model follows the same structure as the sampler above, but using Variables and Ranges
            int nStudents = responsesData.Length;
            int nQuestions = skillsRequired.Length;
            Range student = new Range(nStudents);
            Range question = new Range(nQuestions);
            Range skill = new Range(nSkills);
            var responses = Variable.Array(Variable.Array<bool>(question), student).Named("responses");
            responses.ObservedValue = responsesData;

            pSkill = Variable.Array<double>(skill).Named("pSkill");
            pSkill[skill] = Variable.Beta(1, 1).ForEach(skill);
            slip = Variable.Array<double>(question).Named("slip");
            slip[question] = Variable.Random(slipPrior).ForEach(question);
            guess = Variable.Array<double>(question).Named("guess");
            guess[question] = Variable.Random(guessPrior).ForEach(question);

            hasSkill = Variable.Array(Variable.Array<bool>(skill), student).Named("hasSkill");
            hasSkill[student][skill] = Variable.Bernoulli(pSkill[skill]).ForEach(student);

            VariableArray<int> nSkillsRequired = Variable.Array<int>(question).Named("nSkillsRequired");
            nSkillsRequired.ObservedValue = Util.ArrayInit(nQuestions, q => skillsRequired[q].Length);
            Range skillForQuestion = new Range(nSkillsRequired[question]).Named("skillForQuestion");
            var skillsRequiredForQuestion = Variable.Array(Variable.Array<int>(skillForQuestion), question).Named("skillsRequiredForQuestion");
            skillsRequiredForQuestion.ObservedValue = skillsRequired;
            skillsRequiredForQuestion.SetValueRange(skill);

            using (Variable.ForEach(student))
            {
                using (Variable.ForEach(question))
                {
                    VariableArray<bool> hasSkills = Variable.Subarray(hasSkill[student], skillsRequiredForQuestion[question]);
                    Variable<bool> hasAllSkills = Variable.AllTrue(hasSkills);
                    using (Variable.If(hasAllSkills))
                    {
                        responses[student][question] = !Variable.Bernoulli(slip[question]);
                    }

                    using (Variable.IfNot(hasAllSkills))
                    {
                        responses[student][question] = Variable.Bernoulli(guess[question]);
                    }
                }
            }
        }

        // Sample data from the NIDA model
        public static bool[][] NidaSample(
            int nStudents, 
            int nSkills,
            int[][] skillsRequired,
            Beta slipPrior,
            Beta guessPrior,
            out double[] pSkillOut,
            out double[] slip,
            out double[] guess,
            out bool[][] hasSkill)
        {
            int nQuestions = skillsRequired.Length;
            double[] pSkill = Util.ArrayInit(nSkills, q => Rand.Double());
            slip = Util.ArrayInit(nSkills, s => slipPrior.Sample());
            guess = Util.ArrayInit(nSkills, s => guessPrior.Sample());
            hasSkill = Util.ArrayInit(nStudents, t => Util.ArrayInit(nSkills, s => Rand.Double() < pSkill[s]));
            bool[][] responses = new bool[nStudents][];
            for (int t = 0; t < nStudents; t++)
            {
                responses[t] = new bool[nQuestions];
                for (int q = 0; q < nQuestions; q++)
                {
                    bool[] exhibitsSkill = new bool[skillsRequired[q].Length];
                    for (int s = 0; s < skillsRequired[q].Length; s++)
                    {
                        int skill = skillsRequired[q][s];
                        if (hasSkill[t][skill])
                        {
                            exhibitsSkill[s] = (Rand.Double() > slip[skill]);
                        }
                        else
                        {
                            exhibitsSkill[s] = (Rand.Double() < guess[skill]);
                        }
                    }

                    responses[t][q] = Factor.AllTrue(exhibitsSkill);
                }
            }

            pSkillOut = pSkill;
            return responses;
        }

        // Construct a NIDA model in Infer.NET
        public static void NidaModel(
            bool[][] responsesData, 
            int nSkills, 
            int[][] skillsRequired, 
            Beta slipPrior, 
            Beta guessPrior, 
            out VariableArray<double> pSkill, 
            out VariableArray<double> slip, 
            out VariableArray<double> guess, 
            out VariableArray<VariableArray<bool>, bool[][]> hasSkill)
        {
            // The Infer.NET model follows the same structure as the sampler above, but using Variables and Ranges
            int nStudents = responsesData.Length;
            int nQuestions = responsesData[0].Length;
            Range student = new Range(nStudents);
            Range question = new Range(nQuestions);
            Range skill = new Range(nSkills);
            var responses = Variable.Array(Variable.Array<bool>(question), student).Named("responses");
            responses.ObservedValue = responsesData;

            pSkill = Variable.Array<double>(skill).Named("pSkill");
            pSkill[skill] = Variable.Beta(1, 1).ForEach(skill);
            slip = Variable.Array<double>(skill).Named("slip");
            slip[skill] = Variable.Random(slipPrior).ForEach(skill);
            guess = Variable.Array<double>(skill).Named("guess");
            guess[skill] = Variable.Random(guessPrior).ForEach(skill);

            hasSkill = Variable.Array(Variable.Array<bool>(skill), student).Named("hasSkill");
            hasSkill[student][skill] = Variable.Bernoulli(pSkill[skill]).ForEach(student);

            VariableArray<int> nSkillsRequired = Variable.Array<int>(question).Named("nSkillsRequired");
            nSkillsRequired.ObservedValue = Util.ArrayInit(nQuestions, q => skillsRequired[q].Length);
            Range skillForQuestion = new Range(nSkillsRequired[question]).Named("skillForQuestion");
            var skillsRequiredForQuestion = Variable.Array(Variable.Array<int>(skillForQuestion), question).Named("skillsRequiredForQuestion");
            skillsRequiredForQuestion.ObservedValue = skillsRequired;
            skillsRequiredForQuestion.SetValueRange(skill);

            using (Variable.ForEach(student))
            {
                using (Variable.ForEach(question))
                {
                    VariableArray<bool> hasSkills = Variable.Subarray(hasSkill[student], skillsRequiredForQuestion[question]);
                    VariableArray<double> slipSkill = Variable.Subarray(slip, skillsRequiredForQuestion[question]);
                    VariableArray<double> guessSkill = Variable.Subarray(guess, skillsRequiredForQuestion[question]);
                    VariableArray<bool> exhibitsSkill = Variable.Array<bool>(skillForQuestion).Named("exhibitsSkill");
                    using (Variable.ForEach(skillForQuestion))
                    {
                        using (Variable.If(hasSkills[skillForQuestion]))
                        {
                            exhibitsSkill[skillForQuestion] = !Variable.Bernoulli(slipSkill[skillForQuestion]);
                        }

                        using (Variable.IfNot(hasSkills[skillForQuestion]))
                        {
                            exhibitsSkill[skillForQuestion] = Variable.Bernoulli(guessSkill[skillForQuestion]);
                        }
                    }

                    responses[student][question] = Variable.AllTrue(exhibitsSkill);
                }
            }
        }
    }
}
