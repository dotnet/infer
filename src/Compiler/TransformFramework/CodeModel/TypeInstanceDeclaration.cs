// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// This class models a type declaration which is a specialization of a generic type
    /// declaration
    /// </summary>
    internal class XTypeInstanceDeclaration : ITypeDeclaration
    {
        #region Fields

        private IList<IType> genericArguments;
        private ITypeDeclaration genericType;
        private Type dotNetType; // cached dotnet type

        #endregion

        #region ITypeDeclaration Members

        /// <summary>
        /// Whether the type is abstract - this is the same as for the GenericMethod.
        /// For this reason, it cannot be set in this class
        /// </summary>
        public bool Abstract
        {
            get { return this.genericType.Abstract; }
            set { throw new NotSupportedException(); }
        }

        /// <summary>
        /// This type's base typ - this is the same as for the GenericMethod.
        /// For this reason, it cannot be set in this class
        /// </summary>
        public ITypeReference BaseType
        {
            get
            {
                ITypeReference itr = this.genericType.BaseType;
                if (itr == null)
                    return null;
                else
                    return (ITypeReference) Generics.GetType(itr, this, null);
            }
            set { throw new NotSupportedException(); }
        }

        /// <summary>
        /// Get the specialized collection of fields
        /// </summary>
        public List<IFieldDeclaration> Fields
        {
            get
            {
                List<IFieldDeclaration> fields = this.genericType.Fields;
                List<IFieldDeclaration> instFields = new List<IFieldDeclaration>();

                for (int i = 0; i < fields.Count; i++)
                {
                    IFieldDeclaration ifd = new XFieldDeclaration();
                    ifd.DeclaringType = this;
                    ifd.GenericField = fields[i];
                    ifd.Name = fields[i].Name;
                    ifd.FieldType = Generics.GetType(fields[i].FieldType, this, null);
                    ifd.Initializer = fields[i].Initializer;
                    ifd.Visibility = fields[i].Visibility;
                    ifd.Static = fields[i].Static;
                    ifd.ReadOnly = fields[i].ReadOnly;
                    ifd.Literal = fields[i].Literal;
                    ifd.Attributes.AddRange(fields[i].Attributes);
                    instFields.Add(ifd);
                }
                return instFields;
            }
        }

        /// <summary>
        /// Whether this type is an interface - this is the same as for the GenericMethod.
        /// For this reason, it cannot be set in this class
        /// </summary>
        public bool Interface
        {
            get { return this.genericType.Interface; }
            set { throw new NotSupportedException(); }
        }

        /// <summary>
        /// Get the specialized collection of interfaces implemented by this type
        /// </summary>
        public List<ITypeReference> Interfaces
        {
            get
            {
                List<ITypeReference> interfaces = this.genericType.Interfaces;
                List<ITypeReference> instInterfaces = new List<ITypeReference>();

                for (int i = 0; i < interfaces.Count; i++)
                {
                    ITypeReference itr = (ITypeReference) Generics.GetType(interfaces[i], this, null);
                    instInterfaces.Add(itr);
                }
                return instInterfaces;
            }
        }

        /// <summary>
        /// Get the specialized collection of methods implemented by this class
        /// </summary>
        public List<IMethodDeclaration> Methods
        {
            get
            {
                List<IMethodDeclaration> methods = this.genericType.Methods;
                List<IMethodDeclaration> instMethods = new List<IMethodDeclaration>();

                for (int i = 0; i < methods.Count; i++)
                {
                    IMethodDeclaration instMethod = new XMethodInstanceDeclaration();
                    instMethod.DeclaringType = this;
                    instMethod.GenericMethod = methods[i];
                    for (int j = 0; j < methods[i].GenericArguments.Count; j++)
                    {
                        IType method = Generics.GetType(methods[i].GenericArguments[j], this, null);
                        instMethod.GenericArguments.Add(method);
                    }
                    instMethods.Add(instMethod);
                }
                return instMethods;
            }
        }

        public List<ITypeDeclaration> NestedTypes
        {
            get { throw new Exception("The method or operation is not implemented."); }
        }

        /// <summary>
        /// Get the specialized collection of properties
        /// </summary>
        public List<IPropertyDeclaration> Properties
        {
            get
            {
                List<IPropertyDeclaration> properties = this.genericType.Properties;
                List<IPropertyDeclaration> instProperties = new List<IPropertyDeclaration>();

                for (int i = 0; i < properties.Count; i++)
                {
                    IPropertyDeclaration instProperty = new XPropertyDeclaration();
                    instProperty.DeclaringType = this;
                    instProperty.GenericProperty = properties[i];
                    instProperty.PropertyType = Generics.GetType(properties[i].PropertyType, this, null);
                    instProperty.Name = properties[i].Name;
                    instProperty.SetMethod = properties[i].SetMethod;
                    instProperty.GetMethod = properties[i].GetMethod;
                    instProperty.Attributes.AddRange(properties[i].Attributes);
                    if (properties[i].Parameters != null)
                    {
                        for (int j = 0; j < properties[i].Parameters.Count; j++)
                        {
                            IParameterDeclaration parm = new XParameterDeclaration();
                            parm.Name = properties[i].Parameters[j].Name;
                            parm.ParameterType = Generics.GetType(properties[i].Parameters[j].ParameterType, this, null);
                            parm.Attributes.AddRange(properties[i].Parameters[j].Attributes);
                            instProperty.Parameters.Add(parm);
                        }
                    }
                    instProperties.Add(instProperty);
                }
                return instProperties;
            }
        }

        /// <summary>
        /// Get the specialized collection of events
        /// </summary>
        public List<IEventDeclaration> Events
        {
            get
            {
                List<IEventDeclaration> events = this.genericType.Events;
                List<IEventDeclaration> instEvents = new List<IEventDeclaration>();

                for (int i = 0; i < events.Count; i++)
                {
                    IEventDeclaration ied = new XEventDeclaration();
                    ied.DeclaringType = this;
                    ied.GenericEvent = events[i];
                    ied.Name = events[i].Name;
                    ied.Documentation = events[i].Documentation;
                    ied.EventType = (ITypeReference) Generics.GetType(events[i].EventType, this, null);
                    ied.InvokeMethod = events[i].InvokeMethod;
                    instEvents.Add(ied);
                }
                return instEvents;
            }
        }

        /// <summary>
        /// Whether this type is sealed - this is the same as for the GenericType.
        /// For this reason, it cannot be set in this class
        /// </summary>
        public bool Sealed
        {
            get { return this.genericType.Sealed; }
            set { throw new NotSupportedException(); }
        }

        /// <summary>
        /// Whether this type declaration is partial - this is the same as for the GenericType.
        /// For this reason, it cannot be set in this class
        /// </summary>
        public bool Partial
        {
            get { return this.genericType.Partial; }
            set { throw new NotSupportedException(); }
        }

        /// <summary>
        /// The visibility of this specialization - this is the same as for the GenericMethod.
        /// For this reason, it cannot be set in this class
        /// </summary>
        public TypeVisibility Visibility
        {
            get { return this.genericType.Visibility; }
            set { throw new NotSupportedException(); }
        }

        #endregion

        #region ITypeReference Members

        /// <summary>
        /// The generic type for which this is a specialization.
        /// Setting this property will trigger a resolve of the type
        /// reference to its declaration
        /// </summary>
        public ITypeReference GenericType
        {
            get { return this.genericType; }
            set { this.genericType = value as ITypeDeclaration; }
        }

        /// <summary>
        /// The name of this type - this is the same as for the GenericMethod.
        /// For this reason, it cannot be set in this class
        /// </summary>
        public string Name
        {
            get { return this.genericType.Name; }
            set { throw new NotSupportedException(); }
        }

        /// <summary>
        /// The namespace of this type - this is the same as for the GenericMethod.
        /// For this reason, it cannot be set in this class
        /// </summary>
        public string Namespace
        {
            get { return this.genericType.Namespace; }
            set { throw new NotSupportedException(); }
        }

        /// <summary>
        /// The owner of this specialization - this is the same as for the GenericMethod.
        /// For this reason, it cannot be set in this class
        /// </summary>
        public object Owner
        {
            get { return this.genericType.Owner; }
            set { throw new NotSupportedException(); }
        }

        /// <summary>
        /// Just returns this. It is already a declaration
        /// </summary>
        /// <returns></returns>
        public ITypeDeclaration Resolve()
        {
            return this;
        }

        /// <summary>
        /// Whether this is a value type or not - this is the same as for the GenericMethod.
        /// For this reason, it cannot be set in this class
        /// </summary>
        public bool ValueType
        {
            get { return this.genericType.ValueType; }
            set { throw new NotSupportedException(); }
        }

        #endregion

        #region IComparable Members

        public int CompareTo(object obj)
        {
            ITypeReference itr = obj as ITypeReference;
            if (itr == null)
                return -1;

            return XTypeInstanceReference.Compare(this, itr);
        }

        #endregion

        #region IGenericArgumentProvider Members

        /// <summary>
        /// Generic arguments for this type
        /// </summary>
        public IList<IType> GenericArguments
        {
            get
            {
                if (this.genericArguments == null)
                    this.genericArguments = new List<IType>();
                return this.genericArguments;
            }
        }

        #endregion

        #region ICustomAttributeProvider Members

        /// <summary>
        /// Custom attributes for this declaration
        /// </summary>
        public List<ICustomAttribute> Attributes
        {
            get { return this.genericType.Attributes; }
        }

        #endregion

        #region IDocumentationProvider Members

        /// <summary>
        /// The documentation for this type - this is the same as for the GenericMethod.
        /// For this reason, it cannot be set in this class
        /// </summary>
        public string Documentation
        {
            get { return this.genericType.Documentation; }
            set { throw new NotSupportedException(); }
        }

        #endregion

        #region Object Overrides

        public override string ToString()
        {
            // RTODO
            return this.Name;
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            ITypeReference itr = obj as ITypeReference;
            if (itr == null)
                return false;

            return XTypeInstanceReference.AreEqual(this, itr);
        }

        public override int GetHashCode()
        {
            return this.Name.GetHashCode();
        }

        #endregion Object Overrides

        #region IDotNetType Members

        /// <summary>
        /// Dot net type
        /// </summary>
        public Type DotNetType
        {
            get { return this.dotNetType; }
            set { this.dotNetType = value; }
        }

        #endregion
    }
}