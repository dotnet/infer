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

--Generate a sequence a numbers that later helps in querying
Create table NUMBERS (Num int)
Insert NUMBERS
Select ROW_NUMBER() OVER ( ORDER BY F1)
from JesterJokes.dbo.JokeRating

-- Generate user id yourself as there are no userIDs in the dataset (Each row corresponds to a user)
alter table JesterJokes.dbo.JokeRating add ID int Identity(1,1)
GO
Insert [User]
Select ID from JesterJokes.dbo.JokeRating

-- there are 100 jokes in the dataset, so generate 1...100 item ids
Insert [Item]
Select ROW_NUMBER() OVER (ORDER BY Num)
from NUMBERS
where num <= 100

-- Populate UserItemRating
declare @i as int
declare @query as varchar(1000)
Set @i = 1
while (@i <= 100)
begin
    select @query ='Insert into UserItemRating(UserID, ItemID, Rating) Select ID UserID,' + cast((@i) as varchar(3)) + ' ItemID,F' + cast((@i+1) as varchar(3)) +' Rating from JesterJokes.dbo.JokeRating where F' + cast((@i+1) as varchar(3)) +' <> 99'
    --select @query
    execute(@query)
    set @i = @i + 1
end 

-- There are no features in this dataset so no need to populate UserFeatures and ItemFeatures


select count(*) From Feature
select Count(*) from [User]
select Count(*) from [UserFeature]
select Count(*) from [Item]
select Count(*) from [ItemFeature]
select Count(*) from [UserItemRating]