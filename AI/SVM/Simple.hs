{-# LANGUAGE ScopedTypeVariables, TupleSections, ViewPatterns,
             RecordWildCards, FlexibleInstances #-}
-------------------------------------------------------------------------------
-- |
-- Module     : Bindings.SVM
-- Copyright  : (c) 2011 Ville Tirronen
-- License    : BSD3
--
-- Maintainer : Ville Tirronen <aleator@gmail.com>
--              Paulo Tanimoto <ptanimoto@gmail.com>
--
-- Notes : The module is currently not robust to inputs of wrong dimensionality
--         and is affected by security risks inherent in libsvm model loading.
--
-- Important TODO-items: 
--  * Handle the issue of crashing the system by passing vectors of dimension to the SVMs
--  * Split this library into high and low level parts
--  * Saving and loading SVMs
--
-------------------------------------------------------------------------------
-- This module presents a high level interface for libsvm toolkit. There are three
-- main uses cases for it:
--  
--  1. You have vectors of reals associated with labels and you wish to assign labels
--     to unlabeled vectors. (Classifier machines)
--
--  2. You have a set of vectors and you wish to find similar vectors in a larger set.
--     (One class machines)
--
--  3. You have samples from a function from R^n to R and you wish to estimate that
--     function (Regression machines)
--
--  In case you absolutely need the lower level interface, see AI.Simple.Base or even
--  the bindings-svm packages.

module AI.SVM.Simple (
		 -- * Basic types
                  RegressorType(..), ClassifierType(..)
		 -- * Classifier machines
                 ,trainClassifier, classify   
		 -- * One class machines
                 ,trainOneClass, inSet, OneClassResult(..)
		 -- * Regression machines
                 ,trainRegressor, predictRegression
         -- * Unfortunate utilities
                 ,Persisting(..)
                 )  where

import AI.SVM.Base
import Control.Applicative
import Control.Arrow (second, (***), (&&&))
import Control.Exception
import Control.Monad
import Data.Binary
import Data.Char
import Data.List
import Data.Map (Map)
import Data.Tuple
import System.Directory
import System.IO.Unsafe
import System.Random.MWC
import qualified Data.ByteString.Lazy as B
import qualified Data.Map as Map


-- | Supported SVM classifiers
data ClassifierType =
               C  {cost :: Double}
              | NU {cost :: Double, nu :: Double}

-- | Supported SVM regression machines
data RegressorType =
               Epsilon  Double Double
              | NU_r     Double Double

data SVMClassifier a = SVMClassifier SVM (Map a Double) (Map Double a)
newtype SVMRegressor  = SVMRegressor SVM 
newtype SVMOneClass   = SVMOneClass SVM 

generalizeClassifier C{..} = C_SVC{cost_=cost}
generalizeClassifier NU{..} = NU_SVC{cost_=cost, nu_=nu}

generalizeRegressor (NU_r cost nu)  = NU_SVR{cost_=cost, nu_=nu}
generalizeRegressor (Epsilon cost eps) = EPSILON_SVR{cost_=cost, epsilon_=eps}

-- | A class for things that can be saved to file (i.e. stuff that can't be serialized into memory)
class Persisting a where
    save :: FilePath -> a -> IO ()
    load :: FilePath -> IO a

instance (Ord cls, Binary cls) => Persisting (SVMClassifier cls) where
    save fp (SVMClassifier a to from) = do
        saveSVM fp a
        svm <- B.readFile fp
        B.writeFile fp . encode $ (svm,to,from)
    load fp = do
        (svm,to,from) <- decode <$> B.readFile fp
        r <- withTmp $ \tmp -> do
              B.writeFile tmp svm
              loadSVM tmp
        return $ SVMClassifier r to from

instance Persisting SVMRegressor where
    save fp (SVMRegressor a) = saveSVM fp a
    load fp = SVMRegressor <$> loadSVM fp

instance Persisting SVMOneClass where
    save fp (SVMOneClass a) = saveSVM fp a
    load fp = SVMOneClass <$> loadSVM fp
--
-- * Utilities
randomName = withSystemRandom $ \gen -> map chr <$> replicateM 16 (uniformR (97,122::Int) gen)
                                          :: IO String 

-- | Get a name for a temporary file, run operation with the filename and erase the file if the 
--   operation creates it.
withTmp op = do
        fp <- getTemporaryDirectory
        out <- randomName
        bracket (return ())
                (\() -> do
                         e <- doesFileExist out 
                         when e (removeFile out))
                (\() -> op (fp++"/"++out))

-- | Train an SVM classifier of given type. 
trainClassifier
  :: (SVMVector b, Ord a) =>
     ClassifierType
     -> Kernel
     -> [(a, b)]
     -> (String, SVMClassifier a)

trainClassifier ctype kernel dataset = unsafePerformIO $ do
    let l = zip (nub . labels $ dataset) [1..]
        to   = Map.fromList l
        from = Map.fromList $ map swap l
        doubleDataSet =  map ((\x -> to Map.! x) *** convert) dataset    

    (m,svm) <- trainSVM (generalizeClassifier ctype) kernel doubleDataSet
    return . (m,) $ SVMClassifier svm to from
   where 
    labels = map fst


-- | Classify a vector
classify :: SVMVector v => SVMClassifier a -> v -> a
classify (SVMClassifier svm to from) vector = from Map.! predict svm vector

-- | Train an one class classifier
trainOneClass :: SVMVector a => Double -> Kernel -> [a] -> (String, SVMOneClass)
trainOneClass nu kernel dataset = unsafePerformIO $ do
    let  doubleDataSet =  map (const 1 &&& convert) dataset    

    (m,svm) <- trainSVM (ONE_CLASS nu) kernel doubleDataSet
    return . (m,) $ SVMOneClass svm

-- | The result type of one class svm. The prediction is that point is either `In`the
--   region defined by the training set or `Out`side.
data OneClassResult = Out | In deriving (Eq,Show)

-- | Predict wether given point belongs to the region defined by the oneclass svm
inSet :: SVMVector a => SVMOneClass -> a -> OneClassResult
inSet (SVMOneClass svm) vector = if predict svm vector <0 
                                  then Out
                                  else In

-- | Train an SVM regression machine
trainRegressor
  :: (SVMVector b') =>
     RegressorType -> Kernel -> [(Double, b')] -> (String, SVMRegressor)

trainRegressor rtype kernel dataset = unsafePerformIO $ do
    let  doubleDataSet =  map (second convert) dataset    
    (m,svm) <- trainSVM (generalizeRegressor rtype) kernel doubleDataSet
    return . (m,) $ SVMRegressor svm

-- | Predict value for given vector via regression
predictRegression :: SVMVector a => SVMRegressor -> a -> Double
predictRegression (SVMRegressor svm) (convert -> v) = predict svm v
                         

