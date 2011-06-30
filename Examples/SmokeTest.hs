{-# LANGUAGE ForeignFunctionInterface, BangPatterns, ScopedTypeVariables, TupleSections, 
             RecordWildCards #-}
module Main where

import AI.SVM.Simple
import qualified Data.Vector.Storable as V

main = do
    svm <- loadSVM "model"
    let positiveSample, negativeSample :: V.Vector Double
        positiveSample = V.fromList 
                  [0.708333, 1, 1, -0.320755, -0.105023, -1
                  , 1, -0.419847, -1, -0.225806, 1, -1]
        negativeSample = V.fromList
                  [0.583333 ,-1 ,0.333333 ,-0.603774 ,1 ,-1
                  ,1 ,0.358779 ,-1 ,-0.483871 ,-1 ,1]

    let 
        pos, neg :: Double 
        pos = predict svm positiveSample
        neg = predict svm negativeSample 
    print "Testing a loaded model. Expect (1,-1)."
    print (pos,neg)
    print "Training"
    let trainingData = [(-1, V.fromList [0,1])
                       ,(-1, V.fromList [1,0])
                       ,(1, V.fromList [1,1])
                       ,(1, V.fromList [0,0])
                        ]
    svm2 <- trainSVM (C_SVC 1) Linear trainingData
    print $ predict svm2 $ V.fromList [0,1]
    print $ predict svm2 $ V.fromList [1,0]
    print $ predict svm2 $ V.fromList [0.5,0.5]
    print $ predict svm2 $ V.fromList [1,1]


