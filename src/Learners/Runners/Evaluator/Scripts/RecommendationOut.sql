-- Licensed to the .NET Foundation under one or more agreements.
-- The .NET Foundation licenses this file to you under the MIT license.
-- See the LICENSE file in the project root for more information.

/*******************************************************************************************************************************/
/*Select all ratings of Top N Users ordered by the no. of items they have rated, and then split them into 
    traing and test by a given ratio 
*/


/*
    drop table TestDataExportTop5000
    drop table TrainingDataExportTop5000
*/
    
    drop table TopNUsers
    GO

    declare @NUsers as int
    declare @testSplitSize as float
    declare @testDataSize as int

    select @NUsers = 5000, @testSplitSize = .1

    select * into TopNUsers From 
    (
    select row_number() over (order by count(*) desc) RID, u.userid, count(*) Ratingcount
            from [user] u, UserItemRating r
            where u.UserID = r.UserID
                --and rating > 0
            group by u.userid
    ) UserCounts
    where RID <= @NUsers

    Select @testDataSize = (Select count(*) from useritemrating R, TopNUsers 
                            where TopNUsers.UserID = R.userid
                                    --and rating > 0
                                    ) * @testSplitSize

    select @testDataSize
    --Select top(@testDataSize) 
    --    ValidUserRatings.UserID, ValidUserRatings.ItemID, ValidUserRatings.Rating into UsersTest100
    --From 
    --(
    --    Select * From (select row_number() over(partition by R.userid order by R.userid) RatingID,R.UserID, itemid,rating, Ratingcount
    --                    from  useritemrating R, TopNUsers
    --                    where TopNUsers.UserID = R.userid
    --                    --and rating > 0
    --                    ) NUsersRatings
    --    Where RatingID <= RatingCount * 0.8 --For each user, leave some data explicitly to be in training dataset
    --    --order by UserID, ItemID
    --)ValidUserRatings
    ----order by R.UserID, R.ItemID
    --INNER JOIN
    --(
    --    Select * From
    --    (    
    --        select row_number() over(partition by R.ItemID order by R.ItemID) RatingID,R.UserID, R.itemid, rating, ItemCount
    --        from  useritemrating R, TopNUsers,
    --        (
    --            Select ItemID, Count(*) ItemCount
    --            From UserItemRating R, TopNUsers U
    --            Where R.UserID = U.UserID 
    --            --and rating > 0
    --            Group by ItemID
    --            having Count(*) > 1
    --            --Order by ItemCount desc
    --        ) ItemCounts
    --        where ItemCounts.ItemID = R.ItemID
    --        and TopNUsers.UserID = R.UserID
    --        --and rating > 0
    --    ) NUsersRatings
    --    where NUsersRatings.RatingID <= ItemCount * 0.8   -- For each item, leave some data explicitly in the training dataset
    --) ValidItemsRatings
    --ON ValidUserRatings.itemID = ValidItemsRatings.ItemId AND ValidUserRatings.UserID = ValidItemsRatings.UserID
    --order by NEWID()

    -- Inserting Training data 
    -- The remaining records of top N users goes to training split 

    Select UserID, ItemID, Rating into UserData5000
    From
    (
        select R.UserID, itemid,rating, Ratingcount
        from  useritemrating R, TopNUsers
        where TopNUsers.UserID = R.userid
        --and rating > 0
        --order by ratingcount desc
    ) NUsersRatings
    where not exists (select 1 from UsersTest100 where itemid = NUsersRatings.itemid and userid = NUsersRatings.userid)
    order by NEWID()

    /****************************************************************************************/
    --Test Users: Top N rating for Jester dense subset
    select userid into QryUsers
    From useritemrating where itemid in (5, 7, 8, 13, 15, 16, 17, 18, 19, 20)
    and  userid % 500 = 1
    group by userid
    having count(distinct rating) = 10

    --select UserID, ItemID,Rating --into TopNTestData
    --from useritemrating where userid in (select userid from QryUsers)
    --    select userid
    --    From useritemrating where itemid in (5, 7, 8, 13, 15, 16, 17, 18, 19, 20)
    --    and  userid % 5000 = 1
    --    group by userid
    --    having count(distinct rating) = 10
    --    ) 
    --and itemid in (5, 7, 8, 13, 15, 16, 17, 18, 19, 20)
    --order by userid, rating desc

    --Training Data:
    drop table TopNTraining_3Items_ALL

    select * into TopNTraining_2Items_50K  
    from
    (
    select UserID, ItemID,Cast(Ceiling((rating + 10)/2) as int) Rating   --into TopNTrainingData_12K_2Items 
    from useritemrating where userid not in (select userid from QryUsers)
    and itemid in (5, 7, 8, 13, 15, 16, 17, 18, 19, 20)
    and  userid % 5 = 1
    
    --order by userid, rating desc
    union all
    select r.UserID, r.ItemID,Cast(Ceiling((rating + 10)/2) as int) Rating 
    from useritemrating r, 
        (select top 2 itemid  from (select 5 itemid union select 7 union select 8 union select 13
                                    union select 15 union select 16 union select 17 union select 18 union select 19
                                    union select 20)Dense 
        order by newid()) I
    where r.ItemID = I.itemid
    and r.UserID in (select UserID from QryUsers)
    --order by userid
    )TMP
    order by newid()

    --select userid, count(*)
    --from TopNTraining_5Items_12K
    --group by userid
    --order by count(*)

    select R.userid, itemid, Cast(Ceiling((rating + 10)/2) as int) Rating into TopNGroundTruth
    from UserItemRating R, QryUsers U
    where R.UserID = U.UserID
    and r.ItemID in (5, 7, 8, 13, 15, 16, 17, 18, 19, 20)

    select * from TopNGroundTruth 


    ----Ground truth:
    ----drop table TopNGroundTruth
    --select * into TopNGroundTruth
    --from (
    --select u.id UserID,i.id ItemID,Cast(Ceiling((r.rating + 10)/2) as int) Rating ,ROW_NUMBER() over (partition by u.id order by r.rating desc) Rank   
    --from PerformanceJester.dbo.[TopNTestData] r, PerformanceJester.dbo.[User] u, PerformanceJester.dbo.item i 
    --where r.ItemID = i.ItemID and r.UserID = u.UserID
    --and r.itemid not in (8,15)
    --)TMP 
    --where Rank <=3

    select * from TopNGroundTrut

    bcp "select UserID, ItemId, cast(rating as int)Rating from PerformanceJester.dbo.TopNTraining_2Items_12K order by newid()" queryout C:\TestData\JesterTopN\TopNTraining_2Items_12K.txt  -S. -T -c -t,
    bcp "select UserID, ItemId, cast(rating as int)Rating from PerformanceJester.dbo.TopNGroundTruth" queryout C:\TestData\JesterTopN\TopNGroundTruth.txt  -S. -T -c -t,
    bcp "select UserID from PerformanceJester.dbo.QryUsers" queryout C:\TestData\JesterTopN\QryUsers.txt  -S. -T -c -t,

    /****************************************************************************************/

    --RelatedUsersNGroundTruth:

    select R.userid, itemid, Cast(Ceiling((rating + 10)/2) as int) Rating --into RelatedUsersGroundTruth
    from UserItemRating R
    where r.ItemID in (5, 7, 8, 13, 15, 16, 17, 18, 19, 20)

    bcp "select UserID, ItemId, cast(rating as int)Rating from PerformanceJester.dbo.RelatedUsersGroundTruth order by newid()" queryout C:\TestData\JesterTopN\RelatedUsersGroundTruth.txt  -S. -T -c -t,



    /*************************************************************************************/

    select R.userid, itemid, Cast(Ceiling((rating + 10)/2) as int) Rating into RelatedItemsGroundTruth
    from UserItemRating R
    where r.ItemID in (5, 7, 8, 13, 15, 16, 17, 18, 19, 20)
    and r.userid in (select userid from UserItemRating
                    where ItemID in (5, 7, 8, 13, 15, 16, 17, 18, 19, 20)
                    group by userid
                    having count(*) = 10) 

    bcp "select UserID, ItemId, cast(rating as int)Rating from PerformanceJester.dbo.RelatedItemsGroundTruth order by newid()" queryout C:\TestData\JesterTopN\RelatedItemsGroundTruth.txt  -S. -T -c -t,


    select * from TopNTraining_2Items_ALL
    where rating < 0

    and userid in (select userid from qryusers)
    /**************************************************************************************

    select distinct itemid from RelatedUsersGroundTruth
    order by newid()

    select itemid, count(*) From RelatedUsersGroundTruth
    group by itemid
    having count(*) < 10

    

    /************************************************************************************/
    
    

    select * from topngroundtruth where userid = 16501 order by rating desc
    select * From QryUsers

    select userid, count(*)
    from TopNTraining_5Items_ALL
    where userid in (select userid from qryusers)
    group by userid
    order by count(*)

    update UserFeature set FeatureValue = 1 where FeatureValue ='M'
    update UserFeature set FeatureValue = 2 where FeatureValue ='F'
    select * From feature
    drop table TestData100
    drop table TrainingData100
        


    
    Select 'U,' + D.UserID + ',' + dbo.GetDelimitedUserFeatures(D.UserID,'|') as line-- into TestData100
    From
    (select distinct UserID from TrainingDataExportTop10) D

    Union ALL

    Select 'I,' + D.ItemID + ',' + dbo.GetDelimitedItemFeatures(D.ItemID,'|')
    From
    (select distinct ItemID from TrainingDataExportTop10) D

    Union ALL
    select UserID + ',' + ItemID +',' +Cast( (Cast(Rating as int)) as varchar(5)) 
    From Ratings20K

    select * from feature
    select * From UserItemRating


    select dbo.[GetDelimitedUserFeatures]('"100001"','|')

    select dbo.[GetDelimitedItemFeatures]('"0425182673"','|')

    bcp PerformanceMovieLens.dbo.TrainingData100 out c:\Saleha\MovieLensTraining100.txt  -S. -T -c
    bcp PerformanceMovieLens.dbo.TestData100 out c:\Saleha\MovieLens100.txt  -S. -T -c

    1
101
701
801
1001
select 