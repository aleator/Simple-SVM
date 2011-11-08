{-# LANGUAGE ForeignFunctionInterface, BangPatterns, ScopedTypeVariables, TupleSections, 
             RecordWildCards, NoMonomorphismRestriction #-}
module Main where

import AI.SVM.Simple
import qualified Data.Vector.Storable as V
import Diagrams.Prelude
import Diagrams.Backend.Cairo.CmdLine
import Diagrams.Backend.Cairo



main = do
    let trainingData = [('r', V.fromList [0,0])
                       ,('r', V.fromList [1,1])
                       ,('b', V.fromList [0,1])
                       ,('b', V.fromList [1,0])
                       ,('i', V.fromList [0.5,0.5::Double])
                        ]
    let (m,svm2) = trainClassifier (C 1) (RBF 4) trainingData
    let plot = 
               (circle # fc green # scale 5 )
               `atop` 
               (circle # fc green # scale 5
               `atop` circle # scale 100 # lineWidth 5) # translate (200,200) 
               `atop` 
               (circle # fc green # scale 5 # translate (400,400) )
               `atop` 
               foldl (atop) (circle # scale 1)
               [circle # scale 5 # translate (400*x,400*y) # fc (color svm2 (x,y))
               | x <- [0,0.025..1], y <- [0,0.025..1]] 
    fst $ renderDia Cairo (CairoOptions ("test.png") (PNG (400,400))) plot
  where
    color svm (x,y) = case classify svm  [x,y] of
                        'r' -> red
                        'b' -> blue
                        'i' -> indigo
 
between a x b = a <= x && x <= b
