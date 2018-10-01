---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Tutorials and examples](Infer.NET tutorials and examples.md)

[Page 1](Click model example.md) \| [Page 2](Click Model 1.md) \|  [Page 3](Click model 2.md) \| Page 4

## Click model prediction

This section shows how to run [click model 1](Click Model 1.md) and [click model 2](Click model 2.md) in prediction mode. In fact, we will build a model especially for prediction. This is because the training models we used were structured according to label class, and, for prediction, we typically have no label information. Many of the components of the model are the same as for training, so we will only single out the key differences for discussion.

```csharp
// The observations will be in the form of an array of distributions  
Variable<int> numberOfObservations = Variable.New<int>().Named("NumObs");  
Range r = new Range(numberOfObservations).Named("N");  
VariableArray<Gaussian> observationDistribs = Variable.Array<Gaussian>(r).Named("Obs");  
// Use the marginals from the trained model  
Variable<double> scoreMean = Variable.Random(marginals.marginalScoreMean);  
Variable<double> scorePrec = Variable.Random(marginals.marginalScorePrec);  
Variable<double> judgePrec = Variable.Random(marginals.marginalJudgePrec);  
Variable<double> clickPrec = Variable.Random(marginals.marginalClickPrec);  
Variable<double>[] thresholds = new Variable<double>[numLabels + 1];  
// Variables for each observation  
VariableArray<double> scores = Variable.Array<double>(r).Named("Scores");  
VariableArray<double> scoresJ = Variable.Array<double>(r).Named("ScoresJ");  
VariableArray<double> scoresC = Variable.Array<double>(r).Named("ScoresC");  
scores[r] = Variable.GaussianFromMeanAndPrecision(scoreMean, scorePrec);  
scoresJ[r] = Variable.GaussianFromMeanAndPrecision(scores[r], judgePrec);  
scoresC[r] = Variable.GaussianFromMeanAndPrecision(scores[r], clickPrec);  
// Constrain to the click observation  
Variable.ConstrainEqualRandom(scoresC[r], observationDistribs[r]);  
// The threshold variables  
thresholds[0] = Variable.GaussianFromMeanAndVariance(  
    Double.NegativeInfinity, 0.0);  
for (int i = 1; i < thresholds.Length - 1; i++)  
    thresholds[i] = Variable.Random(marginals.marginalThresh[i]);  
thresholds[thresholds.Length - 1] = Variable.GaussianFromMeanAndVariance(  
    Double.PositiveInfinity, 0.0);  
// Boolean label variables  
VariableArray<bool>[] testLabels = new VariableArray<bool>[numLabels];  
for (int j = 0; j < numLabels; j++)  
{  
    testLabels[j] = Variable.Array<bool>(r).Named("TestLabels" + j);  
    testLabels[j][r] = Variable.IsBetween(scoresJ[r], thresholds[j], thresholds[j + 1]);  
}
```

The first thing to notice is that the inferred variables from the trained model are used as priors for the prediction model. The second point is that the data is not partitioned according to label because we have no labels; consequently, there is no loop over labels. Thirdly, the lower and upper bound thresholds are respectively set to negative infinity, and positive infinity rather than 0.0 and 1.0 - this is so that the label probabilities that will be output by the model sum to 1.0 (**scoreJ**, being Gaussian, will always have some density outside 0.0 and 1.0). Finally, an array of bool variables is set up - the marginals distributions of these (in the form of Bernouilli distributions) will give the probability of each label.

#### Running the prediction

To run the prediction, we first must provide some click data - arrays of click and examination counts. This click data must then be converted into the Gaussian observations (**obs** in the code below) in the same way as the training (though not partitioned by label). This distribution array is then set as the value of the **observationDistrib** parameter, and the marginals are requested from the inference engine.

```csharp
numberOfObservations.ObservedValue = obs.Length;  
observationDistribs.ObservedValue = obs;  
InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());  
Gaussian[] latentScore = Distribution.ToArray<Gaussian[]>(engine.Infer(scores));  
Bernoulli[][] predictedLabels = new Bernoulli[numLabels][];  
for (int j = 0; j < numLabels; j++)  
  predictedLabels[j] = Distribution.ToArray<Bernoulli[]>(engine.Infer(testLabels[j]));
```

#### Interpreting the results

The following table shows some examples run through the model. The Clicks and Exams columns are the input. The score column shows the mean value of the latent score variable, and the label columns show the probabilities of each label. It can be seen that the more examinations there are, the more confident the model is of the labelling. For example, 9 clicks out of 10 is not enough evidence to favour label 2 over label 1, whereas 999 out of 1000 strongly suggests label 2.
```
Clicks   Exams   Score     Label0    Label1    Label2  
10       20      0.4958    0.1827    0.6436    0.1736  
100      200     0.4964    0.1731    0.6620    0.1649  
1000     2000    0.4965    0.1719    0.6643    0.1638  
9        10      0.7929    0.0344    0.4764    0.4891  
99       100     0.9328    0.0095    0.3278    0.6627  
999      1000    0.9489    0.0082    0.3102    0.6816  
10       100     0.1408    0.5767    0.4059    0.0173  
10       1000    0.0522    0.6838    0.3081    0.0081  
10       10000   0.0432    0.6940    0.2985    0.0075
```
#### Model usage

What we have gained from this model is the calibration of human judgement data against click data using query/document pairs for which we have both observations. We can use this either to identify data for which click data and human judgement data are inconsistent and therefore clean up the training data for a ranking model. Or we could use the predicted labels or score to supplement the human judgement training data.

<br/>
[Page 1](Click model example.md) \| [Page 2](Click Model 1.md) \|  [Page 3](Click model 2.md) \| Page 4

​
