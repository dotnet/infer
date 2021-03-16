// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.Serialization;

namespace Microsoft.ML.Probabilistic.Compiler
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Describes the results of a transformation, including errors and warnings encountered.
    /// </summary>
    public class TransformResults
    {
        private List<TransformError> errorsAndWarnings = new List<TransformError>();
        public IReadOnlyList<TransformError> ErrorsAndWarnings { get { return errorsAndWarnings.AsReadOnly(); } }
        protected Dictionary<object, List<TransformError>> errorMap = new Dictionary<object, List<TransformError>>(ReferenceEqualityComparer<object>.Instance);
        public int ErrorCount { get; private set; }
        public int WarningCount { get; private set; }
        public ICodeTransform Transform;

        protected void TransformError(string msg, bool isWarning, Exception exception, object inputElement, object displayTag)
        {
            if (isWarning) WarningCount++;
            else ErrorCount++;
            TransformError te = new TransformError(msg, isWarning, exception, inputElement, displayTag);
            errorsAndWarnings.Add(te);
            List<TransformError> errs;
            if (errorMap.ContainsKey(inputElement))
            {
                errs = errorMap[inputElement];
            }
            else
            {
                errs = new List<TransformError>();
                errorMap[inputElement] = errs;
            }
            errs.Add(te);
        }

        public bool IsSuccess
        {
            get { return ErrorCount == 0; }
        }

        internal void AddWarning(string msg, Exception ex, object inputObject, object tag)
        {
            TransformError(msg, true, ex, inputObject, tag);
        }

        internal void AddError(string msg, Exception ex, object inputObject, object tag)
        {
            TransformError(msg, false, ex, inputObject, tag);
        }

        public override string ToString()
        {
            //string s = "Transform succeeded";
            //if (!IsSuccess) s = "Transform failed";
            StringBuilder sb = new StringBuilder();
            sb.AppendLine(ErrorCount + " error(s) and " + WarningCount + " warning(s):");
            int errors = 0, warnings = 0;
            foreach (var error in errorsAndWarnings)
            {
                if (error.IsWarning)
                {
                    sb.AppendLine("Warning " + (warnings++) + ": " + error);
                }
                else
                {
                    sb.AppendLine("Error " + (errors++) + ": " + error);
                }
            }
            return sb.ToString();
        }

        public bool IsErrors()
        {
            return ErrorCount > 0;
        }

        public void ThrowIfErrors(string msg)
        {
            ThrowIfErrors(msg, false);
        }

        public void ThrowIfErrors(string msg, bool treatWarningsAsErrors)
        {
            int count = ErrorCount;
            if (treatWarningsAsErrors) count += WarningCount;
            if (count > 0)
            {
                TransformFailedException tfe = new TransformFailedException(this, msg + " with " + this);
                tfe.Transform = Transform;
                throw tfe;
            }
        }

        public List<TransformError> GetErrorsForElement(object inputElement)
        {
            if (inputElement == null) return null;
            if (!errorMap.ContainsKey(inputElement)) return null;
            return errorMap[inputElement];
        }
    }

    public class TransformError
    {
        // ErrorText must be a property to show up in the DataGrid.
        public string ErrorText { get; private set; }
        public readonly bool IsWarning;
        public readonly object InputElement;
        public readonly object DisplayTag;
        public readonly Exception exception;
        public const int maxElementStringLength = 1000;

        public TransformError(string msg, bool isWarning, Exception exception, object inputElement, object displayTag)
        {
            this.ErrorText = msg;
            this.IsWarning = isWarning;
            this.exception = exception;
            this.InputElement = inputElement;
            this.DisplayTag = displayTag;
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(ErrorText);
            sb.AppendLine(" in");
            try
            {
                string elementString = InputElement.ToString();
                if (elementString.Length > maxElementStringLength)
                    elementString = elementString.Remove(maxElementStringLength);
                sb.Append(elementString);
            }
            catch (Exception ex)
            {
                sb.Append("Could not evaluate ToString() on inputElement: " + ex);
            }
            if (exception != null)
            {
                sb.AppendLine();
                sb.Append("Details: ");
                sb.Append(exception.Message);
            }
            return sb.ToString();
        }
    }

    [Serializable]
    internal class TransformFailedException : Exception
    {
        public TransformResults Results;
        public ICodeTransform Transform;

        public TransformFailedException(TransformResults tr, string msg)
            : base(msg)
        {
            Results = tr;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="TransformFailedException"/> class.
        /// </summary>
        public TransformFailedException()
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="TransformFailedException"/> class with a specified error message.
        /// </summary>
        /// <param name="message">The error message.</param>
        public TransformFailedException(string message) : base(message)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="TransformFailedException"/> class with a specified error message 
        /// and a reference to the inner exception that is the cause of this exception. 
        /// </summary>
        /// <param name="message">The error message that explains the reason for the exception.</param>
        /// <param name="inner">The exception that is the cause of the current exception.</param>
        public TransformFailedException(string message, Exception inner) : base(message, inner)
        {
        }

        protected TransformFailedException(SerializationInfo info, StreamingContext context) : base(info, context)
        {
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}