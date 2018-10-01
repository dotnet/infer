-- Licensed to the .NET Foundation under one or more agreements.
-- The .NET Foundation licenses this file to you under the MIT license.
-- See the LICENSE file in the project root for more information.

/****** Object:  Table [dbo].[Feature]    Script Date: 16/11/2012 12:10:03 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

SET ANSI_PADDING OFF
GO

CREATE TABLE [dbo].[Feature](
    [FeatureID] [int] IDENTITY(1,1) NOT NULL,
    [FeatureName] [varchar](100) NOT NULL,
 CONSTRAINT [PK_Feature] PRIMARY KEY CLUSTERED 
(
    [FeatureID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]

GO

SET ANSI_PADDING OFF
GO


/****** Object:  Table [dbo].[Item]    Script Date: 16/11/2012 12:10:12 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

SET ANSI_PADDING ON
GO

CREATE TABLE [dbo].[Item](
    [ID] [int] IDENTITY(1,1) NOT NULL,
    [ItemID] [varchar](250) NOT NULL,
 CONSTRAINT [PK_Item] PRIMARY KEY CLUSTERED 
(
    [ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]

GO

SET ANSI_PADDING OFF
GO

/****** Object:  Table [dbo].[ItemFeature]    Script Date: 16/11/2012 12:10:35 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

SET ANSI_PADDING ON
GO

CREATE TABLE [dbo].[ItemFeature](
    [ItemID] [varchar](250) NOT NULL,
    [FeatureID] [int] NOT NULL,
    [FeatureValue] [varchar](max) NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]

GO

SET ANSI_PADDING OFF
GO

/****** Object:  Table [dbo].[User]    Script Date: 16/11/2012 12:10:42 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

SET ANSI_PADDING ON
GO

CREATE TABLE [dbo].[User](
    [ID] [int] IDENTITY(1,1) NOT NULL,
    [UserID] [varchar](250) NOT NULL,
 CONSTRAINT [PK_User] PRIMARY KEY CLUSTERED 
(
    [ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY]

GO

SET ANSI_PADDING OFF
GO

/****** Object:  Table [dbo].[UserFeature]    Script Date: 16/11/2012 12:10:48 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

SET ANSI_PADDING OFF
GO

CREATE TABLE [dbo].[UserFeature](
    [UserID] [varchar](250) NULL,
    [FeatureID] [int] NULL,
    [FeatureValue] [varchar](max) NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]

GO

SET ANSI_PADDING OFF
GO

/****** Object:  Table [dbo].[UserItemRating]    Script Date: 16/11/2012 12:28:31 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

SET ANSI_PADDING ON
GO

CREATE TABLE [dbo].[UserItemRating](
    [UserID] [varchar](250) NULL,
    [ItemID] [varchar](250) NULL,
    [Rating] [decimal](5, 2) NULL
) ON [PRIMARY]

GO

SET ANSI_PADDING OFF
GO


/****** Object:  UserDefinedFunction [dbo].[GetDelimitedItemFeatures]    Script Date: 19/11/2012 11:10:30 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

create function [dbo].[GetDelimitedItemFeatures](@ItemID varchar(250), @delim char)
returns varchar(4000)
as 
begin
    declare @str as varchar(4000)
    Set @str = ''
    Select @str = @str +cast(isnull(FeatureID, -999) as varchar(10)) + ':' + Isnull(FeatureValue , -999) + @delim
                            from [ItemFeature]
                            where [ItemID] like @ItemID 
    return isnull(@str, '')
end


GO


/****** Object:  UserDefinedFunction [dbo].[GetDelimitedUserFeatures]    Script Date: 19/11/2012 11:10:42 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


create function [dbo].[GetDelimitedUserFeatures](@UserID varchar(250), @delim char)
returns varchar(4000)
as 
begin
    declare @str as varchar(4000)
    Set @str = ''
    Select @str = @str +cast(isnull(FeatureID, -999) as varchar(10)) + ':' + Isnull(FeatureValue , -999) + @delim
                            from [UserFeature]
                            where [UserID] like @UserID 
    return isnull(@str, '')
end


GO










