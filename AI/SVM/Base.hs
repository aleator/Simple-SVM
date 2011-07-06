{-# LANGUAGE ForeignFunctionInterface, BangPatterns, ScopedTypeVariables,
             TupleSections, ViewPatterns, RecordWildCards, FlexibleInstances #-}
-------------------------------------------------------------------------------
-- |
-- Module     : Bindings.SVM
-- Copyright  : (c) 2011 Ville Tirronen
-- License    : BSD3
--
-- Maintainer : Ville Tirronen <aleator@gmail.com>
--              Paulo Tanimoto <ptanimoto@gmail.com>
--
-------------------------------------------------------------------------------
-- This module is a medium level interface to libsvm toolkit. 
-- For a high-level description of the C API, refer to the README file 
-- included in the libsvm archive, available for download at 
-- <http://www.csie.ntu.edu.tw/~cjlin/libsvm/>.
--
-- In most cases you should prefer AI.SVM.Simple over this module. AI.SVM.Simple
-- attempts to be slightly more safe and easier to use and exposes almost all of the
-- functionality present here.

module AI.SVM.Base (
                  -- * Types
                   SVM
                 , SVMType(..), Kernel(..)
		 , SVMVector(..)
                 ,getNRClasses
                  -- * File operations
                 ,loadSVM, saveSVM
                  -- * Training
                 ,trainSVM --, crossvalidate
                  -- * Prediction
                 ,predict
                 )  where

import qualified Data.Vector.Storable as V
import qualified Data.Vector as GV
import Data.Vector.Storable ((!))
import Bindings.SVM
import Foreign.C.Types
import Foreign.C.String
import Foreign.Ptr
import Foreign.ForeignPtr
import qualified Foreign.Concurrent as C
import Foreign.Marshal.Utils
import Foreign.Marshal.Array
import Foreign.Marshal.Alloc
import Control.Applicative
import System.IO.Unsafe
import Foreign.Storable
import Control.Monad
import Control.Arrow (first, second, (***), (&&&))
import System.Directory
import Data.IORef
import Control.Exception 
import System.IO.Error
import Data.Tuple
import Data.Map (Map)
import qualified Data.Map as Map
import Data.List

class SVMVector a where
    convert :: a -> V.Vector Double

instance SVMVector (V.Vector Double) where
    convert = id

instance SVMVector (GV.Vector Double) where
    convert = GV.convert

instance SVMVector [Double] where
    convert = V.fromList

instance SVMVector (Double,Double) where
    convert (a,b) = V.fromList [a,b]

instance SVMVector (Double,Double,Double) where
    convert (a,b,c) = V.fromList [a,b,c]

instance SVMVector (Double,Double,Double,Double) where
    convert (a,b,c,d) = V.fromList [a,b,c,d]

instance SVMVector (Double,Double,Double,Double,Double) where
    convert (a,b,c,d,e) = V.fromList [a,b,c,d,e]



{-# SPECIALIZE convertDense :: V.Vector Double -> V.Vector C'svm_node #-}
{-# SPECIALIZE convertDense :: V.Vector Float -> V.Vector C'svm_node #-}
convertDense :: (V.Storable a, Real a) => V.Vector a -> V.Vector C'svm_node
convertDense v = V.generate (dim+1) readVal
    where
        dim = V.length v
        readVal !n | n >= dim = C'svm_node (-1) 0
        readVal !n = C'svm_node (fromIntegral n+1) (realToFrac $ v ! n)

createProblem v = do -- #TODO Check the problem dimension. Libsvm doesn't
                    node_array <- newArray xs
                    class_array <- newArray y
                    offset_array <- newArray $ offsetPtrs node_array
                    return (C'svm_problem (fromIntegral dim) 
                                          class_array 
                                          offset_array
                           ,node_array) 
    where 
        dim = length v
        lengths = map ((+1) . V.length . snd) v
        offsetPtrs addr = take dim 
                          [addr `plusPtr` (idx * sizeOf (C'svm_node undefined undefined)) 
                          | idx <- scanl (+) 0 lengths]
        y   = map (realToFrac . fst)  v
        xs  = concatMap (V.toList . extractSvmNode . snd) v
        extractSvmNode x = convertDense $ V.generate (V.length x) (x !)

deleteProblem (C'svm_problem l class_array offset_array , node_array) =
    free class_array >> free offset_array >> free node_array 


-- | A Support Vector Machine
newtype SVM = SVM  (ForeignPtr C'svm_model)

getModelPtr (SVM fp) = fp

modelFinalizer :: Ptr C'svm_model -> IO ()
modelFinalizer modelPtr = with modelPtr c'svm_free_and_destroy_model

-- | load an svm from a file. This function is rather unsafe, since 
--   a bad model file could cause libsvm to segfault. Also, this could
--   be hugely exploitable by malicious model makers.
loadSVM :: FilePath -> IO SVM
loadSVM fp = do
    e <- doesFileExist fp
    unless e $ ioError $ mkIOError doesNotExistErrorType 
                                   ("Model file "++show fp++" does not exist")
                                   Nothing
                                   (Just fp)
        -- Not finding the file causes a bus error. Could do without that..
    ptr <- withCString fp c'svm_load_model
    let fin = modelFinalizer ptr
    SVM <$> C.newForeignPtr ptr fin

-- | Save an svm to a file.
saveSVM :: FilePath -> SVM -> IO ()
saveSVM fp (getModelPtr -> fptr) = 
    withForeignPtr fptr $ \model_ptr -> 
    withCString fp      $ \cstr      ->
    c'svm_save_model cstr model_ptr

-- | Number of classes the model expects.
getNRClasses (getModelPtr -> fptr) 
    = fromIntegral <$>  withForeignPtr fptr c'svm_get_nr_class

-- | Predict the class of a vector with an SVM.
predict :: (SVMVector a) => SVM -> a -> Double
predict (getModelPtr -> fptr) 
        (convert -> vec) = unsafePerformIO $
                           withForeignPtr fptr $ \modelPtr -> 
                           let nodes = convertDense vec
                           in realToFrac <$> V.unsafeWith nodes 
                                             (c'svm_predict modelPtr)

defaultParamers = C'svm_parameter {
      c'svm_parameter'svm_type = c'C_SVC
    , c'svm_parameter'kernel_type = c'LINEAR
    , c'svm_parameter'degree = 3
    , c'svm_parameter'gamma  = 0.01
    , c'svm_parameter'coef0  = 0
    , c'svm_parameter'cache_size = 100
    , c'svm_parameter'eps = 0.001
    , c'svm_parameter'C   = 1
    , c'svm_parameter'nr_weight = 0
    , c'svm_parameter'weight_label = nullPtr
    , c'svm_parameter'weight       = nullPtr
    , c'svm_parameter'nu = 0.5
    , c'svm_parameter'p  = 0.1
    , c'svm_parameter'shrinking = 1
    , c'svm_parameter'probability = 0
    }

-- | SVM variants
data SVMType = 
               -- | C svm (the default tool for classification tasks)
               C_SVC  {cost_ :: Double}
               -- | Nu svm
             | NU_SVC {cost_ :: Double, nu_ :: Double}
               -- | One class svm
             | ONE_CLASS {nu_ :: Double}
               -- | Epsilon support vector regressor
             | EPSILON_SVR {cost_ :: Double, epsilon_ :: Double}
               -- | Nu support vector regressor 
             | NU_SVR {cost_ :: Double, nu_ :: Double}

-- | SVM kernel type
data Kernel = Linear 
            | Polynomial {gamma :: Double, coef0 :: Double, degree :: Int}
            | RBF {gamma :: Double}
            | Sigmoid {gamma :: Double, coef0 :: Double}
            deriving (Show)

rf = realToFrac
setKernelParameters Linear p = p
setKernelParameters (Polynomial {..}) p = p{c'svm_parameter'gamma=rf gamma
                                           ,c'svm_parameter'coef0=rf coef0
                                           ,c'svm_parameter'degree=fromIntegral degree
                                           ,c'svm_parameter'kernel_type=c'POLY
                                           }
setKernelParameters (RBF {..}) p        = p{c'svm_parameter'gamma=rf gamma 
                                           ,c'svm_parameter'kernel_type=c'RBF
                                           }
setKernelParameters (Sigmoid {..}) p    = p{c'svm_parameter'gamma=rf gamma
                                           ,c'svm_parameter'coef0=rf coef0 
                                           ,c'svm_parameter'kernel_type=c'SIGMOID
                                           }

setTypeParameters (C_SVC cost_) p     = p{c'svm_parameter'C=rf cost_
                                        ,c'svm_parameter'svm_type=c'C_SVC}

setTypeParameters (NU_SVC{..}) p     = p{c'svm_parameter'C=rf cost_
                                        ,c'svm_parameter'nu=rf nu_
                                        ,c'svm_parameter'svm_type=c'NU_SVC}

setTypeParameters (ONE_CLASS{..}) p  = p{c'svm_parameter'nu=rf nu_
                                        ,c'svm_parameter'svm_type=c'ONE_CLASS}

setTypeParameters (EPSILON_SVR{..}) p = p{c'svm_parameter'C=rf cost_
                                        ,c'svm_parameter'p=rf epsilon_
                                        ,c'svm_parameter'svm_type=c'EPSILON_SVR}

setTypeParameters (NU_SVR {..}) p    = p{c'svm_parameter'C=rf cost_
                                        ,c'svm_parameter'nu=rf nu_
                                        ,c'svm_parameter'svm_type=c'NU_SVR}


setParameters svm kernel = parameters
    where 
        parameters = setTypeParameters svm 
                     . setKernelParameters kernel 
                     $ defaultParamers

-- Other params that currently cannot be passed:
-- epsilon -- termination 0.001
-- cachesize -- in mb 100
-- shrinking -- bool 1
-- probability-estimates -- bool 0
-- weights --

foreign import ccall "wrapper"
  wrapPrintF :: (CString -> IO ()) -> IO (FunPtr (CString -> IO ()))

-- |Create an SVM from the training data
trainSVM :: (SVMVector a) => SVMType -> Kernel -> [(Double, a)] -> IO (String, SVM)
trainSVM svm kernel (map (second convert) -> dataSet) = do
    messages <- newIORef []
    let append x = modifyIORef messages (x:)
    pf <- wrapPrintF (peekCString >=> append) 
    c'svm_set_print_string_function pf
    (problem, ptr_nodes) <- createProblem dataSet
    ptr_parameters <- malloc 
    poke ptr_parameters (setParameters svm kernel)
    modelPtr <- with problem $ \ptr_problem -> 
                  c'svm_train ptr_problem ptr_parameters
    message  <- unlines . reverse <$> readIORef messages 
    (message ,) . SVM  <$> C.newForeignPtr modelPtr 
                    (free ptr_parameters
                     >>deleteProblem (problem, ptr_nodes)
                     >>modelFinalizer modelPtr) 

-- |Cross validate SVM. This is faster than training and predicting for each fold
--  separately, since there are no extra conversions done between libsvm and haskell.
--  Currently broken.
crossvalidate
  :: (SVMVector b) => SVMType -> Kernel -> Int -> [(Double, b)] -> IO (String, [Double])
crossvalidate svm kernel folds (map (second convert) -> dataSet) = do
    messages <- newIORef []
    let append x = modifyIORef messages (x:)
    pf <- wrapPrintF (peekCString >=> append) 
          -- The above is just a test. Realistically at that point there
          -- should be an ioref that captures the output which would then
          -- be returned from this function.
    c'svm_set_print_string_function pf
    (problem, ptr_nodes) <- createProblem dataSet
    ptr_parameters <- malloc 
    poke ptr_parameters (setParameters svm kernel)
    
    result_ptr :: Ptr CDouble <- mallocArray (length dataSet)

    with problem $ \ptr_problem -> 
         c'svm_cross_validation ptr_problem ptr_parameters (fromIntegral folds) result_ptr  

    res <- peekArray (length dataSet) result_ptr
    message  <- unlines . reverse <$> readIORef messages 

    free result_ptr >> free ptr_parameters >> deleteProblem (problem,ptr_nodes)

    return (message,map realToFrac res)



