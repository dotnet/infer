---
layout: default
---
[Infer.NET development](../index.md) : [Infer.NET compiler design](../Infer.NET compiler design.md)

## Message transform

Transforms a model specified in terms of channels into the set of message passing operations required to perform inference in that model. This involves:

*   Converting channel variable declarations into a pair of declarations for messages passed forwards and backwards along that channel.
*   Initialising message variables with appropriate marginal prototypes or initial values.
*   Converting each methods call into several operator method calls (typically one for the return value and one for each argument).
*   Removing if statements and creating evidence message operator method calls.

#### Variable declarations

Variable declarations are copied to make forward and backward messages. Only the forward messages are shown, the backward message are identical but with _B instead of _F.

| Input | Output | Notes | 
|------------------------|
| double x; | Gaussian x_F = marginalPrototypeExpression; | |
| double[] xarr; | DistributionArray<Gaussian> xarr_F; | Definition channel |
| double[] x_uses; | Gaussian[] x_uses_F; | Uses channel |
| bool[][] b; | DistributionArray<DistributionArray<Bernoulli>> b_F; | Definition channel |
| bool[][] b_uses; | DistributionArray<Bernoulli>[] b_uses_F; | Uses channel |

Generated variable declarations are marked with `MessageArrayInformation` attributes.

#### Array declarations

Array declarations are converted into DistributionArray declarations (except at the top level of a uses channel).
Only the forward messages are shown, the backward message are identical but with _B instead of _F.

| Input | Output | Notes | 
|------------------------|
| xarr = new double[2]; | xarr_F = new DistributionArray<Gaussian>(marginalPrototypeExpression,2) | Definition channel |
| x_uses = new double[2]; | x_uses_F = ArrayHelper.Fill(new Gaussian[2],marginalPrototypeExpression) | Uses channel |
| b = new bool[10][] | b_F = new DistributionArray<DistributionArray<Bernoulli>(10); | Definition channel |
| b_uses = new bool[10][] | b_uses_F = new DistributionArray<Bernoulli>[10]; | Uses channel |
| b[i] = new bool[sizes[i]] | b_F[i] = new DistributionArray<Bernoulli>(marginalPrototypeExpression,sizes[i]) | Definition channel |
| b_uses[i] = new bool[6]; | b_uses_F[i] = new DistributionArray<Bernoulli>(marginalPrototypeExpression,6); | Uses channel (same as for definition channel in this case) |
| jarray[i] = new bool[sizes[i]]; | b_uses_F[i] = new DistributionArray<Bernoulli>(marginalPrototypeExpression,sizes[i]); | |

#### Methods calls

Static method calls are converted into multiple message operators - typically one for the return value (if any) and one for each argument. Operators are not created for deterministic arguments. The operator method which is called will depend on the original static method and the algorithm being used. Operators assign to backwards messages for arguments and forward messages for return values.
 
Deterministic methods with deterministic arguments are left unchanged.

| Input | Output | Notes | 
|------------------------|
| x=Factor.And(true,false); | x=Factor.And(true,false); | Entirely deterministic |
| y=Factor.Gaussian(m,1.0); | y_F=GaussianOp.SampleAverageConditional(m_F,1.0); <br /> m_B=GaussianOp.MeanAverageConditional(y_B,1.0); | Forward message to y, backward message to m |
| if (a) { <br />   y=Factor.Gaussian(m,1.0); <br /> } | a_B=Bernoulli.FromLogOdds(GaussianOp.LogEvidenceRatio(y_B,m_F,1.0)); <br /> y_F=GaussianOp.SampleAverageConditional(m_F,1.0); <br /> m_B=GaussianOp.MeanAverageConditional(y_B,1.0); | Evidence message |