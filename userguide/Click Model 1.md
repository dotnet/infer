---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Tutorials and examples](Infer.NET tutorials and examples.md)

[Page 1](Click model example.md) \| Page 2 \| [Page 3](Click model 2.md) \| [Page 4](Click Model Prediction.md)

## Click model 1 example

Introductory comments describing the problem are in [click model example](Click model example.md) which should be read first.

#### Representing click observations in the model

In designing this model, we start off by thinking about the nature of the click observations. Each click or non-click provides a bit of evidence about the relevance of the query/document pair. The more examinations we have, the more we believe the evidence. We could think of the set of click/non-click events as the outcome of a binomial experiment - the probability of observing m clicks given N examinations is given by the binomial distribution Bin(m\|N, m) where m is a parameter that we need to infer.

We could include m directly in the model as a parameter to a Binomial factor, but this example adopts a different approach where the posterior for m is calculated outside the model. This posterior can be analytically and simply calculated as a beta distribution. We then use moment-matching to project this distribution onto a Gaussian distribution (the reason for this is that we will later be introducing a Gaussian score variable corresponding to this observation). All of this can be very simply done using the Infer.NET class libraries. For simplicity, we just assume for now that the observation distributions are in a single array, though this will change later.

```csharp
Gaussian[] observationDistrib= new Gaussian[clicks.Length];  
for (int i = 0; i < clicks.Length; i++)  
{    
    int nC = clicks[i];    // Number of clicks int nE = exams[i]; // Number of examinations int nNC = nE - nC; // Number of non-clicks  
    Beta b = new Beta(1.0 + nC, 1.0 + nNC);  
    double m, v;  
    b.GetMeanAndVariance(out m, out v);  
    observationDistrib[i] = Gaussian.FromMeanAndVariance(m, v);  
}
```

We can then imagine, for each query/document pair (index **i**), a Gaussian 'click score' variable **scoresC\[i\]** representing a click probability. The corresponding **observationDistrib\[i\]** represents a distribution over observations for a given query/document pair, and we constrain **scoresC\[i**\] to be a random sample from that distribution. In Infer.NET, the set of click scores and corresponding observations can be indexed by a **Range** object which can be used when specifying this constraint:

```csharp
Range r = new Range("N", observationDistrib);  
...  
Variable.ConstrainEqualRandom(scoresC[r], observationDistrib[r]);
```

#### Partitioning data by judgement label

We will find it useful, when constructing this part of the model, to partition the click data by judgement label. So rather than the single **observationDistrib** array, as shown above, we in fact have three (in this case) arrays - **observationDistribs\[0\],** **observationDistribs\[1**\]**, and** **observationDistribs\[2\]**, indexed by label class. We then make these arrays as observed variable arrays in the inference problem. The ranges are then label dependent, and the **ConstrainEqualRandom** constraint must then be modified to loop over labels:

```csharp
VariableArray<Gaussian>[] observationDistribs = new VariableArray<Gaussian>[numLabels];  
Variable<int>[] numberOfObservations = new Variable<int>[numLabels];  
for (int i = 0; i < numLabels; i++)  
{  
    numberOfObservations[i] = Variable.New<int>().Named("NumObs" + i);  
    Range r = new Range(numberOfObservations[i]).Named("N" + i);  
    observationDistribs[i] = Variable.Array<Gaussian>(r).Named("Obs" + i);  
  
    for (int i = 0; i < numLabels; i++)  
    {  
        ...  
  
        Variable.ConstrainEqualRandom(scoresC[r], observationDistribs[i][r]);  
    }  
}
```

#### Judgement observations

The judgement observations are in the form of labels, and we have no query/document dependent noise. Let's assume for each query/document pair (index **i**), we have a 'judgement score' **scoresJ\[i\]**. We make the modelling assumption that there are thresholds which divide the real line into segments corresponding to the labels, so as to map scores to labels. These thresholds will be variables that we want to learn; let's name these variables **threshold\[i\]**. The stated goal of this model is to reconcile click and judgement data, and we can do this via the scores. In order to put the click probability scores and the human judgement scores on the same footing, we impose a lower-bound threshold of 0.0 and an upper-bound threshold of 1.0 for the judgement scores. This then gives **numLabels+1** thresholds where the top and bottom ones are fixed.

We can then augment the model to include constraints for the **scoresJ** as follows:

```csharp
...  
for (int i = 0; i < numLabels; i++)  
{  
    ...  
    Range r = new Range(numberOfObservations[i]).Named("N" + i);  
    ...  
    Variable.ConstrainBetween(scoresJ[r], thresholds[i], thresholds[i + 1]);  
}
```

#### Linking the score variables

The **scoreC** and **scoreJ** variables are directly or indirectly observed, but are not yet connected to each other in the model. We now link the two observed variables via a score variable **scores\[i\]** indexed by the query/document index. We model **scoresC\[i\]** and **scoresJ\[i\]** as being noisy versions of the same latent variable **scores\[i\]** where the noise is assumed to be Gaussian noise with precisions **clickPrec** and **judgePrec** respectively. **clickPrec** and **judgePrec** are observation-independent, and are unknowns of the problem - i.e. they are random variables whose posterior distributions must be inferred from the observations.

We will also learn the mean of **scores\[i\]** (**scoreMean**), but fix its precision (**scorePrec**) to a nominal point distribution (however, we will keep **scorePrec** as a variable in the model in case we later want to infer it). The model then looks as follows:

```csharp
VariableArray<Gaussian>[] observationDistribs = new VariableArray<Gaussian>[numLabels];  
Variable<int>[] numberOfObservations = new Variable<int>[numLabels];  
for (int i = 0; i < numLabels; i++)  
{  
    numberOfObservations[i] = Variable.New<int>().Named("NumObs" + i);  
    Range r = new Range(numberOfObservations[i], "N" + i);  
    observationDistribs[i] = Variable.Array<Gaussian>(r).Named("Obs" + i);  
    VariableArray<double> scores = Variable.Array<double>(r).Named("Scores" + i);  
    VariableArray<double> scoresJ = Variable.Array<double>(r).Named("ScoresJ" + i);  
    VariableArray<double> scoresC = Variable.Array<double>(r).Named("ScoresC" + i);  
    scores[r] = Variable.GaussianFromMeanAndPrecision(scoreMean, scorePrec);  
    scoresJ[r] = Variable.GaussianFromMeanAndPrecision(scores[r], judgePrec);  
    scoresC[r] = Variable.GaussianFromMeanAndPrecision(scores[r], clickPrec);  
    Variable.ConstrainBetween(scoresJ[r], thresholds[i], thresholds[i + 1]);  
    Variable.ConstrainEqualRandom(scoresC[r], observationDistribs[i][r]);  
}
```

Here we have looped over each label class and built the score variables, factors and constraints for each range of data. However, we have not yet defined the variables which lie outside the ranges - namely **scoreMean**, **scorePrec**, **clickPrec**, **judgePrec**, and **threshold\[\]**. These are the variables that we want to infer. In order to add these to the model, we will first specify their prior distributions.

#### Priors

We use 'conjugate prior' distributions; in a fully factorised Gaussian, the conjugate prior distribution for the mean is itself Gaussian, and the conjugate prior distribution for the precision is a Gamma distribution. Note that we will fix **precScore** to be constant by setting its prior (**priorScorePrec**) to be a point distribution.

We also choose Gaussian distributions for the threshold variables because the **ConstrainBetween** constraint operates on Gaussian distributions. The threshold priors are initialised by evenly splitting the (0,1) interval, and setting the variances to provide some overlap between labels; see the code for details.

```csharp
Gaussian priorScoreMean = Gaussian.FromMeanAndVariance(0.5, 1.0);  
Gamma priorScorePrec = Gamma.FromMeanAndVariance(2.0, 0.0);  
Gamma priorJudgePrec = Gamma.FromMeanAndVariance(2.0, 1.0);  
Gamma priorClickPrec = Gamma.FromMeanAndVariance(2.0, 1.0);  
Gaussian[] priorThreshMean;  
CreateThresholdPriors(numLabels, out priorThresholds);
```

#### The variables to infer

Now the priors are in place, we can specify the random variables that we want to infer. All this code, and the code to specify the prior distributions should go above the model code which references these variables.

```csharp
Variable<double> scoreMean = Variable.Random<double>(priorScoreMean);  
Variable<double> scorePrec = Variable.Random<double>(priorScorePrec);  
Variable<double> judgePrec = Variable.Random<double>(priorJudgePrec);  
Variable<double> clickPrec = Variable.Random<double>(priorClickPrec);  
Variable<double>[] thresholds = new Variable<double>[numLabels + 1];  
for (int i = 0; i < thresholds.Length; i++)  
    thresholds[i] = Variable.Random(priorThresholds[i]);
```

#### Doing the inference

To do the inference, we need to (a) hook up the data, and (b) request the marginals. The following code, which can be found in the Model1 method, does just this:

```csharp
// Get the arrays of human judgement labels, clicks, and examinations  
int[] labels;  
int[] clicks;  
int[] exams;  
LoadData(@"data/ClickModel.txt", false, out labels, out clicks, out exams);  
// Convert the raw click data into uncertain Gaussian observations chunk-by-chunk  
Gaussian[][] allObs = getClickObservations(numLabels, labels, clicks, exams);  
// (a) Set the observation and observation count parameters in the model  
for (int i = 0; i < numLabels; i++)  
{  
    numberOfObservations[i].ObservedValue = allObs[i].Length;  
    observationDistribs[i].ObservedValue = allObs[i];  
}  
// (b) Request the marginals  
//     Inference engine must be EP because of the ConstrainBetween constraint  
InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());  
engine.NumberOfIterations = 10; // Restrict the number of iterations  
ClickModelMarginals marginals = new ClickModelMarginals(numLabels);  
marginals.marginalScoreMean = engine.Infer<Gaussian>(scoreMean);  
marginals.marginalScorePrec = engine.Infer<Gamma>(scorePrec);  
marginals.marginalJudgePrec = engine.Infer<Gamma>(judgePrec);  
marginals.marginalClickPrec = engine.Infer<Gamma>(clickPrec);  
for (int i = 0; i < numThresholds; i++)  
    marginals.marginalThresh[i] = engine.Infer<Gaussian>(thresholds[i]);
```

The first marginal that is requested will trigger the compilation of the model and the inference. **marginals** is just an instance of an application-specific class that has been set up to hold the set of references we are interested in. This information can be passed to a prediction model. The prediction model, results, and their interpretation can be found [here](Click Model Prediction.md).

<br/>
[Page 1](Click model example.md) \| Page 2 \| [Page 3](Click model 2.md) \| [Page 4](Click Model Prediction.md)