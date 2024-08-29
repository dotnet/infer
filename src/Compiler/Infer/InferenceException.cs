using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Probabilistic.Compiler
{
    public class InferenceException : Exception
    {
        public string loopVariable;
        public int loopIndex;
        public int iteration;

        public InferenceException(string loopVariable, int loopIndex, int iteration, Exception exception)
            : base($"Inference failed at iteration {iteration}, {loopVariable}={loopIndex}", exception)
        {
            this.loopVariable = loopVariable;
            this.loopIndex = loopIndex;
            this.iteration = iteration;
        }
    }
}
