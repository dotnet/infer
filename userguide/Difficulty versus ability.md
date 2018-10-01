---
layout: default 
--- 
[Infer.NET user guide](index.md) : [Tutorials and examples](Infer.NET tutorials and examples.md)

## Difficulty versus ability

This example is a model of how people answer questions on a multiple choice test. It explicitly models the trade-off between a person's ability and the difficulty of the question. The model also allows you to estimate the correct answer to each question, which is useful for crowdsourcing and generalizes the approach of majority voting. This model was used in the paper "[How To Grade a Test Without Knowing the Answers --- A Bayesian Graphical Model for Adaptive Crowdsourcing and Aptitude Testing](http://research.microsoft.com/apps/pubs/default.html?id=164692)" by Bachrach et al (ICML 2012), where it was called the DARE model. You can run this example in the [Examples Browser](The examples browser.md). 

In this model, there are multiple subjects who answer multiple questions, each having multiple choices. The data is simply an integer for each subject and question, describing the answer that was chosen. The following variables set this up:

```csharp
int nQuestions = 100;  
int nSubjects = 40;  
int nChoices = 4;  
Range question = new Range(nQuestions);  
Range subject = new Range(nSubjects);  
Range choice = new Range(nChoices);  
var response = Variable.Array(Variable.Array<int>(question), subject);  
response.ObservedValue = data;
```

To explain the data, we introduce four different latent variables. For each subject, we hypothesize a real-valued **ability** variable, where high values increase the subject's probability of answering a question correctly. You can think of this as the subject's level of expertise or concentration on the test. These are assumed to be normally distributed: 

```csharp
Gaussian abilityPrior = new Gaussian(0, 1);  
var ability = Variable.Array<double>(subject);  
ability[subject] = Variable.Random(abilityPrior).ForEach(subject);
```

For each question, we hypothesize a real-valued **difficulty** variable, where high values decrease a subject's probability of answering the question correctly. These are also assumed to be normally distributed: 

```csharp
Gaussian difficultyPrior = new Gaussian(0, 1);  
var difficulty = Variable.Array<double>(question);  
difficulty[question] = Variable.Random(difficultyPrior).ForEach(question);
```

Besides difficulty, a question may have high or low discrimination between people of different abilities. For example, a question that is badly worded may be misinterpreted by a fraction of the subjects, leading to noisy answers regardless of the subject's ability. This is captured by a real-valued **discrimination** variable, where high values increase the effect of a subject's ability. Discrimination is always non-negative. Zero discrimination means that a subject's ability has no effect on whether they will answer the question correctly.

```csharp
Gamma discriminationPrior = Gamma.FromMeanAndVariance(1, 0.01);  
var discrimination = Variable.Array<double>(question);  
discrimination[question] = Variable.Random(discriminationPrior).ForEach(question);
```

Finally, each question has an integer-valued **trueAnswer**. This may be known, as in a classroom scenario, or it may be unknown, as in a crowdsourcing scenario. The model can handle both cases.

```csharp
var trueAnswer = Variable.Array<int>(question);  
trueAnswer[question] = Variable.DiscreteUniform(nChoices).ForEach(question);
```

The generative model now works as follows. For each subject and question, the difference of ability and difficulty is the subject's **advantage** in answering the question correctly. To this advantage we add noise scaled by the discriminatory power of the question. If this noisy advantage is greater than zero, then the subject answers the question correctly, otherwise they choose an answer at random.

```csharp
using (Variable.ForEach(subject)) {  
  using (Variable.ForEach(question)) {  
    var advantage = (ability[subject] - difficulty[question]);  
    var advantageNoisy = Variable.GaussianFromMeanAndPrecision(advantage, discrimination[question]);  
    var correct = (advantageNoisy > 0);  
    using (Variable.If(correct))   
      response[subject][question] = trueAnswer[question];  
    using (Variable.IfNot(correct))    
      response[subject][question] = Variable.DiscreteUniform(nChoices);  
  }  
}
```

To get robust inference in this model, some special settings are necessary, otherwise it tends to generate improper message exceptions. The issue is that the model has highly correlated variables, yet we are using a factorized distribution to approximate it (see the page on [Expectation Propagation](Expectation Propagation.md)). This leads to slow and unstable convergence. To help convergence we instruct the scheduler to process subjects sequentially, so that all variables are updated after each subject, i.e. 40 times per iteration, rather than once per iteration. A nice benefit of these settings is that the inference converges rather quickly (less than 5 iterations).

```csharp
InferenceEngine engine = new InferenceEngine();  
engine.NumberOfIterations = 5;  
subject.AddAttribute(new Sequential()); // needed to get stable convergence
question.AddAttribute(new Sequential()); // needed to get stable convergence
```

To test the inference under this model, we generate a data set from known parameters and compare the learned parameters to the true ones. Notice that the Sample method has the same structure as the Infer.NET model. This happens because the Infer.NET model essentially is a sampler but expressed using the Infer.NET primitives instead of C#. The results are shown below. The estimated parameters are pretty good. 

```
99% TrueAnswers correct  
difficulty[0] = Gaussian(1.914, 0.3346) (sampled from 2.4)  
difficulty[1] = Gaussian(-0.2033, 0.08233) (sampled from -0.24)  
difficulty[2] = Gaussian(-0.341, 0.0806) (sampled from -0.21)  
difficulty[3] = Gaussian(-0.03086, 0.08715) (sampled from 0.26)  
discrimination[0] = Gamma(101.1, 0.00994)[mean=1.005] (sampled from 1)  
discrimination[1] = Gamma(104.1, 0.0096)[mean=0.9995] (sampled from 1.1)  
discrimination[2] = Gamma(104.4, 0.009614)[mean=1.004] (sampled from 1.1)  
discrimination[3] = Gamma(103.6, 0.009661)[mean=1.001] (sampled from 0.87)  
ability[0] = Gaussian(0.2524, 0.03589) (sampled from 0.58)  
ability[1] = Gaussian(0.5612, 0.03622) (sampled from 0.81)  
ability[2] = Gaussian(1.335, 0.04739) (sampled from 1.5)  
ability[3] = Gaussian(0.171, 0.03551) (sampled from 0.33)
```

Note that if the ability parameters are all equal, then the estimate of the true answers will be identical to majority voting, since the most likely true answer will be the answer that most subjects chose. Thus to compare the results of this model to majority voting, just set the ability parameters to a constant. If you do this on this dataset, only 97% of the estimated trueAnswers are correct. Thus the ability parameters help to do better vote aggregation.

#### How to handle missing data

The provided code assumes that every subject has answered every question. If this is not the case, then some changes are necessary. One approach is to leave the response array unobserved and apply constraints to the individual elements that were observed. Another approach is use conditionals to skip over the missing elements, as explained in [How to handle missing data](How to handle missing data.md). However both of these are inefficient. The most efficient approach is to restructure the data as a collection of (subject, question, response) observations. Instead of looping over all subjects and questions, you only loop over the provided observations. The model becomes:

```csharp
Range obs = new Range(nObservations);  
var subjectOfObs = Variable.Array<int>(obs);  
subjectOfObs.ObservedValue = ...;  
var questionOfObs = Variable.Array<int>(obs);  
questionOfObs.ObservedValue = ...;  
var response = Variable.Array<int>(obs);  
response.ObservedValue = ...;  
using (Variable.ForEach(obs)) {  
var q = questionOfObs[obs];  
  var advantage = (ability[subjectOfObs[obs]] - difficulty[q]);  
  var advantageNoisy = Variable.GaussianFromMeanAndPrecision(advantage, discrimination[q]);  
  var correct = (advantageNoisy > 0);  
using (Variable.If(correct))  
response[obs] = trueAnswer[q];  
using (Variable.IfNot(correct))  
response[obs] = Variable.DiscreteUniform(nChoices);  
}
```

For an example of this approach, see the [forum](http://social.microsoft.com/Forums/en-US/infer.net/thread/de3b14da-7818-4fbe-95dd-552649f16c3a).

You can find another example of using Infer.NET for crowdsourcing in [Community-Based Bayesian Classifier Combination](Community-Based Bayesian Classifier Combination.md).​​​​