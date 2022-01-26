// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;

namespace ClinicalTrial
{
    public class ClinicalTrialModel
    {
        //--------------------------------------------
        // Observed variables
        //--------------------------------------------
        public Variable<int> numberPlacebo;
        public Variable<int> numberTreated;
        public VariableArray<bool> placeboGroupOutcomes;
        public VariableArray<bool> treatedGroupOutcomes;

        //--------------------------------------------
        // To be inferred
        //--------------------------------------------
        public Variable<bool> isEffective;
        public Variable<double> probIfPlacebo;
        public Variable<double> probIfTreated;

        //--------------------------------------------
        // Inferences
        //--------------------------------------------
        public Bernoulli posteriorTreatmentIsEffective;
        public Beta posteriorProbIfPlacebo;
        public Beta posteriorProbIfTreated;

        // The inference engine
        public InferenceEngine engine = new InferenceEngine { };

        public ClinicalTrialModel()
        {
            numberPlacebo = Variable.New<int>().Named("numberPlacebo");
            numberTreated = Variable.New<int>().Named("numberTreated");
            Range P = new Range(numberPlacebo);
            Range T = new Range(numberTreated);
            placeboGroupOutcomes = Variable.Observed<bool>(new bool[] { true }, P).Named("placeboGroupOutcomes");
            treatedGroupOutcomes = Variable.Observed<bool>(new bool[] { true }, T).Named("treatedGroupOutcomes");

            isEffective = Variable.Bernoulli(0.2).Named("isEffective");
            using (Variable.If(isEffective))
            {
                // Model if treatment is effective
                probIfPlacebo = Variable.Beta(1, 1).Named("probIfPlacebo");
                placeboGroupOutcomes[P] = Variable.Bernoulli(probIfPlacebo).ForEach(P);
                probIfTreated = Variable.Beta(1, 1).Named("probIfTreated");
                treatedGroupOutcomes[T] = Variable.Bernoulli(probIfTreated).ForEach(T);
            }

            using (Variable.IfNot(isEffective))
            {
                // Model if treatment is not effective
                Variable<double> probAll = Variable.Beta(1, 1).Named("probAll");
                placeboGroupOutcomes[P] = Variable.Bernoulli(probAll).ForEach(P);
                treatedGroupOutcomes[T] = Variable.Bernoulli(probAll).ForEach(T);
            }

            engine.Compiler.WriteSourceFiles = false;
            engine.Compiler.GenerateInMemory = true;
        }

        public void Infer(bool[] treated, bool[] placebo)
        {
            // Set the observed values
            numberPlacebo.ObservedValue = placebo.Length;
            numberTreated.ObservedValue = treated.Length;
            placeboGroupOutcomes.ObservedValue = placebo;
            treatedGroupOutcomes.ObservedValue = treated;

            // Infer the hidden values
            posteriorTreatmentIsEffective = engine.Infer<Bernoulli>(isEffective);
            posteriorProbIfPlacebo = engine.Infer<Beta>(probIfPlacebo);
            posteriorProbIfTreated = engine.Infer<Beta>(probIfTreated);
        }
    }
}
