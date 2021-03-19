// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// This class models a type declaration. It is not the specialization of a generic type
    /// but may be a generic type definition
    /// </summary>
    internal class XTypeDeclaration : ITypeDeclaration
    {
        #region Fields

        private List<ICustomAttribute> attributes;
        private List<IFieldDeclaration> fields;
        private List<IMethodDeclaration> methods;
        private List<IPropertyDeclaration> properties;
        private List<IEventDeclaration> events;
        private IList<IType> genericArguments;
        private List<ITypeDeclaration> nestedTypes;
        private ITypeReference baseType;
        private List<ITypeReference> interfaces;
        private int bitFlags;
        private object owner;
        private string name;
        private string nmspace;
        private string documentation;
        // Values for bitFlags - first three bit
        private const int interfaceFlag = 0x0001;
        private const int abstractFlag = 0x0002;
        private const int sealedFlag = 0x0004;
        private const int partialFlag = 0x0008;
        private TypeVisibility typeVis = TypeVisibility.Public;
        private Type dotNetType; // cached dotnet type

        #endregion

        #region Object overrides

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            ITypeDeclaration type = obj as ITypeDeclaration;
            if (type == null)
                return false;

            if (Name != type.Name ||
                Namespace != type.Namespace ||
                !Util.AreEqual(BaseType, type.BaseType))
                return false;

            bool deepEquality = false;
            if (!deepEquality) return true;
            if (Interface != type.Interface ||
                Abstract != type.Abstract ||
                Sealed != type.Sealed ||
                Partial != type.Partial)
                return false;
            if (!EnumerableExtensions.AreEqual(GenericArguments, type.GenericArguments)) return false;
            if (!EnumerableExtensions.AreEqual(Attributes, type.Attributes)) return false;
            if (!EnumerableExtensions.AreEqual(Fields, type.Fields)) return false;
            if (!EnumerableExtensions.AreEqual(Properties, type.Properties)) return false;
            if (!EnumerableExtensions.AreEqual(Methods, type.Methods)) return false;
            if (!EnumerableExtensions.AreEqual(NestedTypes, type.NestedTypes)) return false;
            return true;
        }

        public override int GetHashCode()
        {
            int hash = Hash.Start;
            hash = Hash.Combine(hash, Name.GetHashCode());
            hash = Hash.Combine(hash, Namespace.GetHashCode());
            hash = Hash.Combine(hash, (BaseType == null) ? 0 : BaseType.GetHashCode());
            return hash;
        }

        /// <summary>
        /// ToString override
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            //ILanguageWriter writer = new CSharpWriter() as ILanguageWriter;
            //return writer.TypeDeclarationSource(this);
            StringBuilder sb = new StringBuilder();
            switch (Visibility)
            {
                case TypeVisibility.Private:
                    sb.Append("private ");
                    break;
                default:
                    sb.Append("public ");
                    break;
            }
            if (Interface)
            {
                sb.Append("interface ");
            }
            else
            {
                if (Abstract)
                    sb.Append("abstract ");
                sb.Append("class ");
            }
            sb.Append(Name);
            sb.Append(" { ... }");
            return sb.ToString();
        }

        #endregion

        #region ITypeDeclaration Members

        /// <summary>
        /// Whether this is an abstract type or not
        /// </summary>
        public bool Abstract
        {
            get { return ((this.bitFlags & abstractFlag) != 0); }
            set { this.bitFlags = (this.bitFlags & ~abstractFlag) | (!value ? 0 : abstractFlag); }
        }

        /// <summary>
        /// Type that this type is inherited from
        /// </summary>
        public ITypeReference BaseType
        {
            get { return this.baseType; }
            set { this.baseType = value; }
        }

        /// <summary>
        /// Fields declared by this type
        /// </summary>
        public List<IFieldDeclaration> Fields
        {
            get
            {
                if (this.fields == null)
                    this.fields = new List<IFieldDeclaration>();
                return this.fields;
            }
        }

        /// <summary>
        /// Whether this type is an interface or not
        /// </summary>
        public bool Interface
        {
            get { return ((this.bitFlags & interfaceFlag) != 0); }
            set { this.bitFlags = (this.bitFlags & ~interfaceFlag) | (!value ? 0 : interfaceFlag); }
        }

        /// <summary>
        /// Interfaces that this type derives from
        /// </summary>
        public List<ITypeReference> Interfaces
        {
            get
            {
                if (this.interfaces == null)
                    this.interfaces = new List<ITypeReference>();
                return this.interfaces;
            }
        }

        /// <summary>
        /// Methods declared by this type
        /// </summary>
        public List<IMethodDeclaration> Methods
        {
            get
            {
                if (this.methods == null)
                    this.methods = new List<IMethodDeclaration>();
                return this.methods;
            }
        }

        /// <summary>
        /// Declared types nested in this type
        /// </summary>
        public List<ITypeDeclaration> NestedTypes
        {
            get
            {
                if (this.nestedTypes == null)
                    this.nestedTypes = new List<ITypeDeclaration>();
                return this.nestedTypes;
            }
        }

        /// <summary>
        /// Properties declared in this type
        /// </summary>
        public List<IPropertyDeclaration> Properties
        {
            get
            {
                if (this.properties == null)
                    this.properties = new List<IPropertyDeclaration>();
                return this.properties;
            }
        }

        /// <summary>
        /// Events declared in this type
        /// </summary>
        public List<IEventDeclaration> Events
        {
            get
            {
                if (this.events == null)
                    this.events = new List<IEventDeclaration>();
                return this.events;
            }
        }

        /// <summary>
        /// Whether this type is sealed or not
        /// </summary>
        public bool Sealed
        {
            get { return ((this.bitFlags & sealedFlag) != 0); }
            set { this.bitFlags = (this.bitFlags & ~sealedFlag) | (!value ? 0 : sealedFlag); }
        }

        /// <summary>
        /// Whether this declaration is partial or not
        /// </summary>
        public bool Partial
        {
            get { return ((this.bitFlags & partialFlag) != 0); }
            set { this.bitFlags = (this.bitFlags & ~partialFlag) | (!value ? 0 : partialFlag); }
        }

        /// <summary>
        /// Visibility of this type
        /// </summary>
        public TypeVisibility Visibility
        {
            // This makes use of the fact that TypeVisibility.NestedFamilyOrAssembly(=7) is bitwise
            // or of all Type visibilities
            get { return typeVis; }
            set { this.typeVis = value; }
        }

        #endregion

        #region ITypeReference Members

        /// <summary>
        /// TypeDeclaration is not used to represent specializations - use TypeInstanceDeclaration
        /// instead. This method will return null on get and throw an exception on set
        /// </summary>
        public ITypeReference GenericType
        {
            get { return null; }
            set { throw new NotSupportedException("Use TypeInstanceDeclaration type to represent a specialization of a generic type"); }
        }

        /// <summary>
        /// Name of the type
        /// </summary>
        public string Name
        {
            get { return this.name; }
            set { this.name = value; }
        }

        /// <summary>
        /// Namespace of the type
        /// </summary>
        public string Namespace
        {
            get { return this.nmspace; }
            set { this.nmspace = value; }
        }

        /// <summary>
        /// Owner of the type
        /// </summary>
        public object Owner
        {
            get { return this.owner; }
            set { this.owner = value; }
        }

        /// <summary>
        /// Resolve. As this is a declaration, it resolves to itself
        /// </summary>
        /// <returns></returns>
        public ITypeDeclaration Resolve()
        {
            return this;
        }

        public bool ValueType
        {
            get
            {
                // RTODO
                throw new Exception("The method or operation is not implemented.");
            }
            set
            {
                // This is not implemented
                throw new Exception("The method or operation is not implemented.");
            }
        }

        #endregion

        #region IComparable Members

        public int CompareTo(object obj)
        {
            ITypeReference reference = obj as ITypeReference;
            if (reference == null)
            {
                return -1;
            }
            return CompareItems.CompareTypeReferences(this, reference);
        }

        #endregion

        #region IGenericArgumentProvider Members

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
        /// Attributes attached to this type
        /// </summary>
        public List<ICustomAttribute> Attributes
        {
            get
            {
                if (this.attributes == null)
                {
                    this.attributes = new List<ICustomAttribute>();
                }
                return this.attributes;
            }
        }

        #endregion

        #region IDocumentationProvider Members

        /// <summary>
        /// Documentation for this type
        /// </summary>
        public string Documentation
        {
            get { return this.documentation; }
            set { this.documentation = value; }
        }

        #endregion

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