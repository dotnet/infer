// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tutorials
{
    [Example("Applications", "Difficulty versus ability in test taking")]
    public class DifficultyAbility
    {
        public void Run()
        {
            InferenceEngine engine = new InferenceEngine();
            if (!(engine.Algorithm is Algorithms.ExpectationPropagation))
            {
                Console.WriteLine("This example only runs with Expectation Propagation");
                return;
            } 
            
            Rand.Restart(0);

            int nQuestions = 100;
            int nSubjects = 40;
            int nChoices = 4;
            Gaussian abilityPrior = new Gaussian(0, 1);
            Gaussian difficultyPrior = new Gaussian(0, 1);
            Gamma discriminationPrior = Gamma.FromMeanAndVariance(1, 0.01);

            double[] trueAbility, trueDifficulty, trueDiscrimination;
            int[] trueTrueAnswer;
            int[][] data = Sample(
                nSubjects,
                nQuestions,
                nChoices,
                abilityPrior,
                difficultyPrior,
                discriminationPrior,
                out trueAbility,
                out trueDifficulty,
                out trueDiscrimination,
                out trueTrueAnswer);
            Range question = new Range(nQuestions).Named("question");
            Range subject = new Range(nSubjects).Named("subject");
            Range choice = new Range(nChoices).Named("choice");
            var response = Variable.Array(Variable.Array<int>(question), subject).Named("response");
            response.ObservedValue = data;

            var ability = Variable.Array<double>(subject).Named("ability");
            ability[subject] = Variable.Random(abilityPrior).ForEach(subject);
            var difficulty = Variable.Array<double>(question).Named("difficulty");
            difficulty[question] = Variable.Random(difficultyPrior).ForEach(question);
            var discrimination = Variable.Array<double>(question).Named("discrimination");
            discrimination[question] = Variable.Random(discriminationPrior).ForEach(question);
            var trueAnswer = Variable.Array<int>(question).Named("trueAnswer");
            trueAnswer[question] = Variable.DiscreteUniform(choice).ForEach(question);

            using (Variable.ForEach(subject))
            {
                using (Variable.ForEach(question))
                {
                    var advantage = (ability[subject] - difficulty[question]).Named("advantage");
                    var advantageNoisy = Variable.GaussianFromMeanAndPrecision(advantage, discrimination[question]).Named("advantageNoisy");
                    var correct = (advantageNoisy > 0).Named("correct");
                    using (Variable.If(correct))
                    {
                        response[subject][question] = trueAnswer[question];
                    }

                    using (Variable.IfNot(correct))
                    {
                        response[subject][question] = Variable.DiscreteUniform(choice);
                    }
                }
            }

            engine.NumberOfIterations = 5;
            subject.AddAttribute(new Models.Attributes.Sequential());  // needed to get stable convergence
            question.AddAttribute(new Models.Attributes.Sequential());  // needed to get stable convergence
            bool doMajorityVoting = false; // set this to 'true' to do majority voting
            if (doMajorityVoting)
            {
                ability.ObservedValue = Util.ArrayInit(nSubjects, i => 0.0);
                difficulty.ObservedValue = Util.ArrayInit(nQuestions, i => 0.0);
                discrimination.ObservedValue = Util.ArrayInit(nQuestions, i => 1.0);
            }

            var trueAnswerPosterior = engine.Infer<IList<Discrete>>(trueAnswer);
            int numCorrect = 0;
            for (int q = 0; q < nQuestions; q++)
            {
                int bestGuess = trueAnswerPosterior[q].GetMode();
                if (bestGuess == trueTrueAnswer[q])
                {
                    numCorrect++;
                }
            }

            double pctCorrect = 100.0 * numCorrect / nQuestions;
            Console.WriteLine("{0}% TrueAnswers correct", pctCorrect.ToString("f0"));
            var difficultyPosterior = engine.Infer<IList<Gaussian>>(difficulty);
            for (int q = 0; q < System.Math.Min(nQuestions, 4); q++)
            {
                Console.WriteLine("difficulty[{0}] = {1} (sampled from {2})", q, difficultyPosterior[q], trueDifficulty[q].ToString("g2"));
            }

            var discriminationPosterior = engine.Infer<IList<Gamma>>(discrimination);
            for (int q = 0; q < System.Math.Min(nQuestions, 4); q++)
            {
                Console.WriteLine("discrimination[{0}] = {1} (sampled from {2})", q, discriminationPosterior[q], trueDiscrimination[q].ToString("g2"));
            }

            var abilityPosterior = engine.Infer<IList<Gaussian>>(ability);
            for (int s = 0; s < System.Math.Min(nSubjects, 4); s++)
            {
                Console.WriteLine("ability[{0}] = {1} (sampled from {2})", s, abilityPosterior[s], trueAbility[s].ToString("g2"));
            }
        }

        public int[][] Sample(
            int nSubjects,
            int nQuestions,
            int nChoices,
            Gaussian abilityPrior,
            Gaussian difficultyPrior,
            Gamma discriminationPrior,
            out double[] ability,
            out double[] difficulty,
            out double[] discrimination,
            out int[] trueAnswer)
        {
            ability = Util.ArrayInit(nSubjects, s => abilityPrior.Sample());
            difficulty = Util.ArrayInit(nQuestions, q => difficultyPrior.Sample());
            discrimination = Util.ArrayInit(nQuestions, q => discriminationPrior.Sample());
            trueAnswer = Util.ArrayInit(nQuestions, q => Rand.Int(nChoices));
            int[][] response = new int[nSubjects][];
            for (int s = 0; s < nSubjects; s++)
            {
                response[s] = new int[nQuestions];
                for (int q = 0; q < nQuestions; q++)
                {
                    double advantage = ability[s] - difficulty[q];
                    double noise = Gaussian.Sample(0, discrimination[q]);
                    bool correct = (advantage > noise);
                    if (correct)
                    {
                        response[s][q] = trueAnswer[q];
                    }
                    else
                    {
                        response[s][q] = Rand.Int(nChoices);
                    }
                }
            }

            return response;
        }
    }
}
