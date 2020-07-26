namespace TestFSharp

open System.Collections.Generic
open Microsoft.ML.Probabilistic
open Microsoft.ML.Probabilistic.FSharp
open Microsoft.ML.Probabilistic.Models
open Microsoft.ML.Probabilistic.Utilities
open Microsoft.ML.Probabilistic.Distributions
open Microsoft.ML.Probabilistic.Math

module DifficultyAbility =
    let main() = 
        Rand.Restart(0);
        let nQuestions = 100
        let nSubjects = 40
        let nChoices = 4
        let abilityPrior = Gaussian(0.0, 1.0)
        let difficultyPrior = Gaussian(0.0, 1.0)
        let discriminationPrior = Gamma.FromMeanAndVariance(1.0, 0.01)

        let Sample(nSubjects:int,nQuestions:int,nChoices:int,abilityPrior:Gaussian,difficultyPrior:Gaussian,discriminationPrior:Gamma)=
          let ability= Util.ArrayInit( nSubjects, (fun _-> abilityPrior.Sample()))
          let difficulty = Util.ArrayInit(nQuestions, (fun _ -> difficultyPrior.Sample()))
          let discrimination = Util.ArrayInit(nQuestions, (fun _ -> discriminationPrior.Sample()))
          let trueAnswer = Util.ArrayInit(nQuestions, (fun _ -> Rand.Int(nChoices)))
          let response:int[][] = Array.zeroCreate nSubjects 
          for s in 0..(nSubjects-1) do
            response.[s] <- Array.zeroCreate nQuestions
            for q in 0..(nQuestions-1) do
              let advantage = ability.[s] - difficulty.[q]
              let noise = Gaussian.Sample(0.0, discrimination.[q])
              let correct = (advantage > noise)
              if (correct) then
                  response.[s].[q] <- trueAnswer.[q]
              else
                  response.[s].[q] <- Rand.Int(nChoices)
          (response, ability,difficulty,discrimination,trueAnswer)


        let data,trueAbility,trueDifficulty,trueDiscrimination,trueTrueAnswer = Sample(nSubjects,nQuestions,nChoices,abilityPrior,difficultyPrior,discriminationPrior)

        let question = Range(nQuestions).Named("question")
        let subject = Range(nSubjects).Named("subject")
        let choice = Range(nChoices).Named("choice")

        //let response = Variable.Array(Variable.Array<int>(question), subject).Named("response")
        let response = Variable.Array<VariableArray<int>, int [][]>(Variable.Array<int>(question), subject).Named("response")

        response.ObservedValue <- data

        let ability = Variable.Array<double>(subject).Named("ability")
        Variable.ForeachBlock subject ( fun s ->  ability.[s] <- Variable.Random(abilityPrior) )
        let difficulty = Variable.Array<double>(question).Named("difficulty")
        Variable.ForeachBlock question ( fun q ->  difficulty.[q] <- Variable.Random(difficultyPrior) )
        let discrimination = Variable.Array<double>(question).Named("discrimination")
        Variable.ForeachBlock question ( fun q ->  discrimination.[q] <- Variable.Random(discriminationPrior) )
        let trueAnswer = Variable.Array<int>(question).Named("trueAnswer")
        Variable.ForeachBlock question ( fun q ->  trueAnswer.[q] <- Variable.DiscreteUniform(choice) )

        Variable.ForeachBlock subject (fun s -> 
          Variable.ForeachBlock question (fun q -> 
            let advantage = (ability.[s] - difficulty.[q]).Named("advantage")
            let advantageNoisy = Variable.GaussianFromMeanAndPrecision(advantage, discrimination.[q]).Named("advantageNoisy")
            let correct = (advantageNoisy >> 0.0).Named("correct")
            Variable.IfBlock correct (fun _->response.[s].[q] <- trueAnswer.[q]) (fun _->response.[s].[q] <- Variable.DiscreteUniform(choice))
            ()
          )
        )

        let engine = InferenceEngine()
        engine.NumberOfIterations <- 5
        subject.AddAttribute(Models.Attributes.Sequential())
        question.AddAttribute(Models.Attributes.Sequential())
        let doMajorityVoting = false; // set this to 'true' to do majority voting
        if doMajorityVoting then
            ability.ObservedValue <- Util.ArrayInit(nSubjects, (fun i -> 0.0))
            difficulty.ObservedValue <- Util.ArrayInit(nQuestions, (fun i -> 0.0))
            discrimination.ObservedValue <- Util.ArrayInit(nQuestions, (fun i -> 0.0))

        let trueAnswerPosterior = engine.Infer<IReadOnlyList<Discrete>>(trueAnswer)

        let mutable numCorrect = 0
        for q in 0..(nQuestions-1) do
            let bestGuess = trueAnswerPosterior.[q].GetMode()
            if (bestGuess = trueTrueAnswer.[q]) then
              numCorrect<-numCorrect+1

        let pctCorrect:float = 100.0 * (float numCorrect) / (float nQuestions)
        printfn "%f TrueAnswers correct" pctCorrect

        let difficultyPosterior = engine.Infer<IReadOnlyList<Gaussian>>(difficulty)


        for q in 0..(System.Math.Min(nQuestions, 4)-1) do
          printfn "difficulty[%i] = %A (sampled from %f)"  q  difficultyPosterior.[q] trueDifficulty.[q]

        let discriminationPosterior = engine.Infer<IReadOnlyList<Gamma>>(discrimination)
        for q in 0..(System.Math.Min(nQuestions, 4)-1) do
          printfn "discrimination[%i] = %A (sampled from %f)"  q  discriminationPosterior.[q] trueDiscrimination.[q]

        let abilityPosterior = engine.Infer<IReadOnlyList<Gaussian>>(ability)
        for s in 0..(System.Math.Min(nQuestions, 4)-1) do
          printfn "ability[%i] = %A (sampled from %f)"  s  abilityPosterior.[s] trueAbility.[s]

