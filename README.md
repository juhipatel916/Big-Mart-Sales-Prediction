# Big-Mart-Sales-Prediction
Sales prediction using supervised learning- Decision Tree and Random Forest Algorithms as a part of Data Science course work with data preprossessing using hypothesis generation, data exploration, data cleaning, feature engineering, model creation using skicit learn python’s ML library(non parallel approach) and also using Apache hadoop and Apache Mahout environment(parallel approach) and  measured efficiency in both .

We have implemented using two approaches. 
1. Non-Parallel approach using python sklearn library 
2. Parallel approch using apache mahout 

Input Train File: train.csv 
Input Test File: test.csv  

1. Non-Parallel approach using python sklearn library: 

   Run "Big Mart Sales Prediction using classification.py" which will create output Files with sales prediction.
   After Data Cleaning Modified Input Files:train_modified.csv
                                            test_modified.csv
   Output Files: decisionTree1.csv
                 decisionTree2.csv
                 randomForest1.csv
                 randomForest2.csv  

2. Parallel approch using apache mahout 

       * Create cluster with apache maven and apache mahout on the top of apache hadoop and JDK. 
       * Put the modified input files on HDFS. 
       * Prepare the description file that describe type of variables using mahout-core-0.5-job.jar with following command: 
         mahout describe –p /input_data/train.csv –f /input_data/in.info –d I 4 N C 32 N L 
       * Split data into Train and Test set by specifying percentage for each of them. 
         mahout splitDataset --input /input_data/train.csv –output /output_data --trainingPercentage 0.7 --probePercentage 0.3 
       * Build model using mahout-examples-0.11.0-job.jar which has the implementation of Machine Learning algorithms in mapreduce for Trainset. 
         mahout buildforest –d /output_data/trainingSet/* -ds /input_data/in.info –sl 3 –p –t 10 –o /output_model 
       * Test Model for Test set. 
         mahout testforest –i /output_data/probeSet –ds /input_data/in.info –m /output_model –a –mr –o /output_prediction                      
