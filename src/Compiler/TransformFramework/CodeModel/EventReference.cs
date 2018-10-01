// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Event reference
    /// </summary>
    public class XEventReference : IEventReference
    {
        #region Fields

        private ITypeReference eventType;
        private IEventReference genericEvent;
        private IType declaringType;
        private string name;
        private WeakReference declaration;

        #endregion

        #region IEventReference Members

        /// <summary>
        /// Event type reference
        /// </summary>
        public ITypeReference EventType
        {
            get { return this.eventType; }
            set { this.eventType = value; }
        }

        /// <summary>
        /// Gets/sets the generic event reference. This will be specialised by
        /// the containing specialised class
        /// </summary>
        public IEventReference GenericEvent
        {
            get { return this.genericEvent; }
            set { this.genericEvent = value; }
        }

        /// <summary>
        /// Resolve the event reference
        /// </summary>
        /// <returns></returns>
        public IEventDeclaration Resolve()
        {
            if (this.declaration == null || !this.declaration.IsAlive)
            {
                List<IEventDeclaration> declarations = (this.DeclaringType as ITypeReference).Resolve().Events;
                string name = this.name;
                foreach (IEventDeclaration ifd in declarations)
                {
                    if (name == ifd.Name && this.EventType.Equals(ifd.EventType))
                    {
                        this.declaration = new WeakReference(ifd);
                        return (IEventDeclaration) this.declaration.Target;
                    }
                }
                return null;
            }
            else
                return (IEventDeclaration) this.declaration.Target;
        }

        #endregion

        #region IMemberReference Members

        /// <summary>
        /// Declaring type for this property
        /// </summary>
        public IType DeclaringType
        {
            get { return this.declaringType; }
            set { this.declaringType = value; }
        }

        /// <summary>
        /// Name of this property
        /// </summary>
        public string Name
        {
            get { return this.name; }
            set { this.name = value; }
        }

        #endregion

        #region IComparable Members

        /// <summary>
        /// Compare this event reference to another object
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public int CompareTo(object obj)
        {
            IEventReference reference = obj as IEventReference;
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

        #region Object Overrides

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return this.Name;
        }

        /// <summary>
        ///
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            IEventReference reference = obj as IEventReference;
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
        /// 
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            return this.Name.GetHashCode();
        }

        #endregion Object Overrides
    }
}