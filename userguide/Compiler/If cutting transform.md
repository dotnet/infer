---
layout: default
---
[Infer.NET user guide](../index.md) : [Infer.NET development](../Infer.NET development.md) : [Infer.NET compiler design](../Infer.NET compiler design.md)

## If cutting transform

This transform ensures that 'if' statements with stochastic conditions contain exactly one statement. This is needed to get the correct usage counts for evidence variables (each statement that sends evidence is one use of that evidence variable). The transform ensures this by finding an 'if' statement with a stochastic condition and cuts it across the contained statements, so that each statement is in its own if statement. Note that GateTransform has already ensured that 'if' statements with stochastic conditions are inside of 'for' loops, so we do not need to worry about splitting 'for' loops. GateTransform has also ensured that 'if' statements with stochastic conditions are inside of 'if' statements with constant conditions, so we do not need to split those either.
 
If a stochastic variable is declared and assigned in the same line, then this is split into two lines, each placed in its own 'if' statement. This is necessary because the declaration will generate a variable factor that sends its own evidence message. 
 
For nested 'if' statements with stochastic conditions, only the innermost if statement is retained.

The transform removes 'if' statements with stochastic conditions from around statements that do not send evidence messages. This includes:

*   Non-stochastic declarations and assignments
*   Array creation statements (optional - done to avoid dummy uses of evidence variables)
*   Infer() statements.
*   Declarations and assignments for variables with the `DoNotSendEvidence` attribute.

The transform requires that the input 'if' statements with stochastic conditions do not have else clauses.

| **Input** | **Output** |
|------------------------|
| `if (a) {` <br /> `double prec = 1.0;` <br /> `double x;` <br /> `double y = Factor.Random(prior);` <br /> `bool[] barray = new bool[4];` <br /> `x = Factor.Gaussian(y,prec);` <br /> `InferNet.Infer(x);` <br /> `}` | `double prec = 1.0;` <br /> `if (a) {` <br /> `double x;` <br /> `}` <br /> `if (a) {` <br /> `double y;` <br /> `}` <br /> `if (a) {` <br /> `y = Factor.Random(prior);` <br /> `}` <br /> `bool[] barray = new bool[4];` <br /> `if (a) {` <br /> `x = Factor.Gaussian(y,1);` <br /> `}` <br /> `InferNet.Infer(x);` |
| `if (a) {` <br /> `double x;` <br /> `if (b) {` <br /> `double y = Factor.Random(prior);` <br /> `}` <br /> `}` | `if (a) {` <br /> `double x;` <br /> `}` <br /> `if (b) {` <br /> `double y;` <br /> `}` <br /> `if (b) {` <br /> `y = Factor.Random(prior);` <br /> `}` |