-- Licensed to the .NET Foundation under one or more agreements.
-- The .NET Foundation licenses this file to you under the MIT license.
-- See the LICENSE file in the project root for more information.
/*
delete from useritemrating
delete from itemfeature
delete from UserFeature
delete from item
delete from [user]
delete From feature
*/

Insert into Feature  Select 'Age'
Insert into Feature  Select 'Location'
Insert into Feature  Select 'BookTitle'
Insert into Feature  Select 'BookAuthor'
Insert into Feature  Select 'YearOfPublication'
Insert into Feature  Select 'Publisher'

Insert into [User] 
select distinct UserID from BookRating.dbo.[BX-Users]

Insert into Item
select distinct B.["ISBN"] from BookRating.dbo.[BX-Books] B

--Populate User and features
Insert  UserFeature (UserID, FeatureID, FeatureValue)
Select    U.UserID,
        F.FeatureID,
        (case   when FeatureName = 'Age' then U.Age
                when FeatureName = 'Location' then U.Location
        end)
from  Feature F, BookRating.dbo.[BX-Users] U
where FeatureName in ('Age', 'Location')

--Populate items and features
Insert ItemFeature 
Select B.["ISBN"], 
    F.FeatureID,
    (case when FeatureName = 'YearOfPublication' then B.["Year-Of-Publication"]
            when FeatureName = 'Publisher' then B.["Publisher"]
            when FeatureName = 'BookAuthor' then B.["Book-Author"]
            when FeatureName = 'BookTitle' then B.["Book-Title"]
    end)
from  Feature F, BookRating.dbo.[BX-Books] B
where FeatureName in ('BookAuthor', 'BookTitle', 'YearOfPublication', 'Publisher')

--Populate ratings
Insert UserItemRating(UserID,ItemID, Rating)
Select R.UserID, R.ISBN, cast( substring(R.BookRating,2,len(R.BookRating) - 2) as decimal(5,2))
from  BookRating.dbo.[BX-Book-Ratings] R

select count(*) From Feature
select Count(*) from [User]
select Count(*) from [UserFeature]
select Count(*) from [Item]
select Count(*) from [ItemFeature]
select Count(*) from [UserItemRating]
