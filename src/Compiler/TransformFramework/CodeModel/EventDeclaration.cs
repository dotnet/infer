// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Event declaration
    /// </summary>
    public class XEventDeclaration : IEventDeclaration
    {
        #region Fields

        private List<ICustomAttribute> attributes;
        private IEventReference genericEvent;
        private IType declaringType;
        private ITypeReference eventType;
        private IMethodReference invokeMethod;
        private string name;
        private string documentation;

        #endregion

        #region IEventDeclaration Members

        /// <summary>
        /// Reference for event invoke method
        /// </summary>
        public IMethodReference InvokeMethod
        {
            get { return this.invokeMethod; }
            set { this.invokeMethod = value; }
        }

        #endregion

        #region IEventReference Members

        /// <summary>
        /// Type of the event
        /// </summary>
        public ITypeReference EventType
        {
            get { return this.eventType; }
            set { this.eventType = value; }
        }

        /// <summary>
        /// The generic event from which this event is derived (if any)
        /// </summary>
        public IEventReference GenericEvent
        {
            get { return this.genericEvent; }
            set { this.genericEvent = value; }
        }

        /// <summary>
        /// Resolve - just returns this as it is already a declaration
        /// </summary>
        /// <returns></returns>
        public IEventDeclaration Resolve()
        {
            return this;
        }

        #endregion

        #region IMemberReference Members

        /// <summary>
        /// Declaring type of this event
        /// </summary>
        public IType DeclaringType
        {
            get { return this.declaringType; }
            set { this.declaringType = value; }
        }

        /// <summary>
        /// The name of this event
        /// </summary>
        public string Name
        {
            get { return this.name; }
            set { this.name = value; }
        }

        #endregion

        #region IComparable Members

        /// <summary>
        /// Compares this instance with another object
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public int CompareTo(object obj)
        {
            IEventDeclaration reference = obj as IEventDeclaration;
            if (reference == null)
            {
                throw new NotSupportedException();
            }
            int ret = this.DeclaringType.CompareTo(reference.DeclaringType);
            if (0 == ret)
                ret = String.Compare(this.Name, reference.Name, StringComparison.InvariantCulture);
            if (0 == ret)
                ret = this.EventType.CompareTo(reference.EventType);
            return ret;
        }

        #endregion

        #region ICustomAttributeProvider Members

        /// <summary>
        /// The custome attributes attached to this event
        /// </summary>
        public List<ICustomAttribute> Attributes
        {
            get
            {
                if (this.attributes == null)
                    this.attributes = new List<ICustomAttribute>();
                return this.attributes;
            }
        }

        #endregion

        #region IDocumentationProvider Members

        /// <summary>
        /// The documentation for this event
        /// </summary>
        public string Documentation
        {
            get { return this.documentation; }
            set { this.documentation = value; }
        }

        #endregion

        #region Object Overrides

        /// <summary>
        /// String representation
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            ILanguageWriter writer = new CSharpWriter() as ILanguageWriter;
            return writer.EventDeclarationSource(this);
        }

        /// <summary>
        /// Determines whether this instance is equal to another instance
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            IEventDeclaration reference = obj as IEventDeclaration;
            if (reference == null)
                return false;

            if (this.DeclaringType.Equals(reference.DeclaringType) &&
                this.Name.Equals(reference.Name) &&
                this.EventType.Equals(reference.EventType))
                return true;
            else
                return false;
        }

        /// <summary>
        /// Hash code for the instance
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            return this.Name.GetHashCode() + this.EventType.GetHashCode();
        }

        #endregion Object Overrides
    }
}