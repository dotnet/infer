-- Licensed to the .NET Foundation under one or more agreements.
-- The .NET Foundation licenses this file to you under the MIT license.
-- See the LICENSE file in the project root for more information.
//
Insert into Feature  Select 'Genre'
Insert into Feature  Select 'Gender'
Insert into Feature  Select 'Age'
Insert into Feature  Select 'Occupation'
Insert into Feature  Select 'Zip'


Insert into [User] 
select distinct UserID from MovieLens.dbo.Users

Insert into Item
select distinct MovieID from MovieLens.dbo.movies

--Populate ratings
Insert UserItemRating(UserID,ItemID, Rating)
Select UserID, MovieID, Rating
from  MovieLens.dbo.ratings

--Populate User and features
Insert  UserFeature (UserID, FeatureID, FeatureValue)
Select U.UserID,
        F.FeatureID,
        (case   when FeatureName = 'Gender' then U.Gender
                when FeatureName = 'Age' then U.Age
                when FeatureName = 'Occupation' then U.Occupation
                when FeatureName = 'Zip' then U.Zip
        end)
from  Feature F, MovieLens.dbo.users U 
where FeatureName in ('Gender', 'Age', 'Occupation','Zip')

--Populate items and features
Insert ItemFeature 
Select M.MovieID,
    F.FeatureID,
    (case when FeatureName = 'Genre' then M.Genre
    end)
from  Feature F, MovieLens.dbo.movies M
where FeatureName in ('Genre')

select count(*) From Feature
select Count(*) from [User]
select Count(*) from [UserFeature]
select Count(*) from [Item]
select Count(*) from [ItemFeature]
select Count(*) from [UserItemRating]



select * From Feature
select * from [User]
select * from [UserFeature]
select * from [Item]
select * from [ItemFeature]
select * from [UserItemRating]

