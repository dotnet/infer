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

-- 'A' prefix correponds to Attribute line ( i.e. item)
insert into item 
select [Column 1] From MSWeb.dbo.[anonymous-msweb]
where [Column 0] = 'A'

Insert into Feature Select 'Title'
Insert into Feature Select 'URL'

--delete from ItemFeature
Insert into ItemFeature
Select [Column 1], 
        F.FeatureID,
        [Column 3]
From MSWeb.dbo.[anonymous-msweb] W, Feature F
where [Column 0] = 'A'
and FeatureName in ('Title', 'URL')

update ItemFeature 
set FeatureValue = substring(FeatureValue, 2,charindex('"',FeatureValue,2 )-2)
From Feature Where ItemFeature.FeatureID = Feature.FeatureID
and FeatureName = 'Title'

update ItemFeature 
set FeatureValue = substring(FeatureValue, charindex('"',FeatureValue,2 ), len(FeatureValue))
From Feature Where ItemFeature.FeatureID = Feature.FeatureID
and FeatureName = 'URL'

-- 'C' prefix correponds to a case for each user
Insert into [User]
select distinct [Column 2] 
From MSWeb.dbo.[anonymous-msweb] W
where [Column 0] = 'C'

alter table MSWeb.dbo.[anonymous-msweb]  add ID int identity(1,1)

--Multiple V lines after a C line represent Votes (i.e. urls visited ) by that cases
Insert into UserItemRating (UserID, ItemID, Rating)
Select Cases.CaseID, W1.[Column 1] ,1
From 
    MSWeb.dbo.[anonymous-msweb] W1,
    (select W1.ID RowID, W1.[Column 2] as CaseID,
        (Select min(ID) from MSWeb.dbo.[anonymous-msweb] W2 where W2.[ID] > W1.ID and W2.[Column 0] = 'C') NextCaseID
    From MSWeb.dbo.[anonymous-msweb] W1
    where W1.[Column 0] = 'C'
    --order by RowID
    ) Cases 
where W1.[Column 0] = 'V' and W1.ID between Cases.RowID and ISNULL(Cases.NextCaseID, 9999999)
order by Cases.CaseID

select count(*) From Feature
select Count(*) from [User]
select Count(*) from [UserFeature]
select Count(*) from [Item]
select Count(*) from [ItemFeature]
select Count(*) from [UserItemRating]
 



 