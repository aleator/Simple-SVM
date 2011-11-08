{-# LANGUAGE ForeignFunctionInterface, BangPatterns, ScopedTypeVariables, TupleSections, 
             RecordWildCards, NoMonomorphismRestriction #-}
module Main where

import AI.SVM.Simple
import qualified Data.Vector.Storable as V
import Diagrams.Prelude
import Diagrams.Backend.Cairo.CmdLine
import Diagrams.Backend.Cairo
import System.Random.MWC
import Control.Applicative
import Control.Monad


scaledN g = (+0.5) . (/10)  <$> normal g

main = do
    pts ::[(Double,Double)] 
        <- withSystemRandom $ \g -> zip <$> replicateM 30 (scaledN g :: IO Double)
                                        <*> replicateM 30 (scaledN g :: IO Double)
    let (msg, svm2) = trainOneClass 0.01 (RBF 1) pts
    putStrLn msg
    let plot = 
               foldl (atop) (circle # scale 0.025)
               [circle # scale 0.022 # translate (x,y) # fc green
               | (x,y) <- pts ] 
               `atop` 
               foldl (atop) (circle # scale 0.025)
               [circle # scale 0.012 # translate (x,y) # fc (color svm2 (x,y))
               | x <- [0,0.025..1], y <- [0,0.025..1]] 
    fst $ renderDia Cairo (CairoOptions ("test.png") (PNG (400,400))) (plot # lineWidth 0.002)
  where
    color svm pt = case inSet svm pt of 
                    In  -> red
                    Out -> black
 
between a x b = a <= x && x <= b
