# Import libraries
import findspark
findspark.init()

from pyspark.mllib.recommendation import *
import random
from operator import *
from collections import defaultdict
import math

# Initialize Spark Context
from pyspark import SparkContext
from pyspark import SparkConf
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

# Import test files from location into RDD variables
artistData = sc.textFile("data_raw/artist_data_small.txt")
artistAlias = sc.textFile("data_raw/artist_alias_small.txt")
userArtistData = sc.textFile("data_raw/user_artist_data_small.txt")

# userArtistData:  userid      artistid        playcount
# ArtistData:      artistid    artist_name
# Artist Alias     badid       goodid

# Testing
#print("\nArtist Data \n",end="");print(artistData.take(5))

# Split by  \t and convert id to int
artistData = artistData.map(lambda x: tuple([int(x.split('\t')[0]),x.split('\t')[1]]))
# Split by \t or ' ' and convert to int
artistAlias = artistAlias.map(lambda x: tuple([int(num_in_str) for num_in_str in x.split('\t')]))
userArtistData = userArtistData.map(lambda x:tuple([int(num_in_str) for num_in_str in x.split(' ')] ))

# Testing
# print("\nArtist data \n",end=""); print(artistData.take(2))

# Create a dictionary of the 'artistAlias' dataset
artistAliasDict = artistAlias.collectAsMap()

# If artistid exists, replace with artistsid from artistAlias, else retain original
def correctArtistIds(x):
    # get(key,def_val) returns value if key is present in dict; default value otherwise
    mod = artistAliasDict.get(x[1],x[1])
    return (x[0],mod,x[2])
userArtistData = userArtistData.map(correctArtistIds)
# Testing
#print("\nUpdated User Artist Data \n",end="");print(userArtistData.take(2))

# Create an RDD consisting of 'userid' and 'playcount' objects of original tuple
uidpc = userArtistData.map(lambda x: [x[0],x[2]])

# Testing
#print("\nUID PC RDD \n",end="");print(uidpc.take(2))

# Count instances by key and store in broadcast variable

# total_uidpc is  (uid,( sum_of_playcount, #entries ))
total_uidpc = uidpc.aggregateByKey((0,0),  lambda a,b: (a[0] + b,    a[1] + 1),
                                           lambda a,b: (a[0] + b[0], a[1] + b[1]))
# Testing
#print("\ntotal_uidpc \n",end=""); print(total_uidpc.take(2))

# Compute and display users with the highest playcount along with their mean playcount across artists
# rdd_result is (total,(uid,avg))   ---- total as key allows us to take by max total
rdd_result = total_uidpc.map(lambda x: (x[1][0],(x[0],x[1][0]/x[1][1])))

# Testing
#print("\nresult_rdd \n",end=""); print(rdd_result.top(3))

# Split the 'userArtistData' dataset into training, validation and test datasets. Store in cache for frequent access
trainData, validationData, testData = userArtistData.randomSplit([4,4,2],13)
trainData.cache()
validationData.cache()
testData.cache()

def modelEval(model, dataset):
    # All artists in the 'userArtistData' dataset
    allArtists = userArtistData.map(lambda x: x[1]).distinct()
   
    # Set of all users in the current (Validation/Testing) dataset
    users = set(dataset.map(lambda x: x[0]).collect())
    
    # Create a dictionary of (user, [artist_ids]) for current (Validation/Testing) dataset
    currDict = dataset.map(lambda x: (x[0],x[1])).distinct().groupByKey().collectAsMap()
    
    # Create a dictionary of (userid, [artist_ids]) for training dataset
    trainDict = trainData.map(lambda x: (x[0],x[1])).groupByKey().collectAsMap()
    
    # For each user, calculate the prediction score i.e. similarity between predicted and actual artists
    modelScore = []
    for user in users:
        
        userRdd = sc.parallelize([user])
        # Do not predict on artists in training data - better than filtering artists after prediction
        artistsNotInTrain = allArtists.filter(lambda x: x not in list(trainDict[user]))
        cartesian = userRdd.cartesian(artistsNotInTrain)        # (user,a1),(user,a2)...(user,artistsNotInTrain)
        
        # Predict artist ratings
        predicted = model.predictAll(cartesian)
        
        # Calculate X from training data for this user
        X = len(list(currDict[user]))
        #X = len(list(trainDict[user]))
        
        # Select top X results from prediction
        # Rating is key here => allows sort by rating
        top_predicted = predicted.map(lambda r: (r[2],r[1])).top(X)
        
        # score - Intersection of predicted artists and artists in Validation/testing dataset for that user
        count = 0
        artist_list = list(currDict[user])
        total = len(artist_list)
        for tup in top_predicted:
            if tup[1] in artist_list:
                count+=1
        modelScore.append(count/total)
        
    # Print average score of the model for all users for the specified rank
    final = 0
    for score in modelScore:
        final += score
    print("The model score for rank "+str(rank)+" is ~"+str(final/len(modelScore)))

rankList = [2,10,20]
for rank in rankList:
    model = ALS.trainImplicit(trainData, rank , seed=345)
    modelEval(model,validationData)

bestModel = ALS.trainImplicit(trainData, rank=2, seed=345)
rank = 2 # Required since I am printing rank in modelEval so Rank must be set to correct var 
modelEval(bestModel, testData)

# Find the top 5 artists for a particular user and list their names
recommended = bestModel.recommendProducts(1059637,5)
al = artistData.collectAsMap()
for i,rec in enumerate(recommended):
    print("Artist "+str(i)+": "+al[rec[1]])

