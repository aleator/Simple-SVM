{-# LANGUAGE ForeignFunctionInterface, BangPatterns, ScopedTypeVariables, TupleSections, 
             RecordWildCards #-}
module Main where

import SVMSimple
import qualified Data.Vector.Storable as V

main = do
    svm <- loadSVM "model"
    let positiveSample = V.fromList 
                  [0.708333, 1, 1, -0.320755, -0.105023, -1
                  , 1, -0.419847, -1, -0.225806, 1, -1]
        negativeSample = V.fromList
                  [0.583333 ,-1 ,0.333333 ,-0.603774 ,1 ,-1
                  ,1 ,0.358779 ,-1 ,-0.483871 ,-1 ,1]

    let 
        pos = predict svm positiveSample
        neg = predict svm negativeSample 
    print "Testing a loaded model. Expect (1,-1)."
    print (pos,neg)
    print "Training"
    let trainingData = [(-1, V.fromList [0])
                       ,(-1, V.fromList [20])
                       ,(1, V.fromList [21])
                       ,(1, V.fromList [50])
                        ]
    svm2 <- trainSVM (C_SVC 1) Linear trainingData
    print $ predict svm2 $ V.fromList [0]
    print $ predict svm2 $ V.fromList [19]
    print $ predict svm2 $ V.fromList [12]
    print $ predict svm2 $ V.fromList [40]


