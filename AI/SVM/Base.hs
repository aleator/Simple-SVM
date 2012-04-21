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
                 , SVMNodes
                   
                 ,getNRClasses
                  -- * File operations
                 ,loadSVM, saveSVM
                  -- * Training
                 ,trainSVM, crossvalidate
                  -- * Prediction
                 ,predict
                 )  where

import AI.SVM.Common
import Bindings.SVM
import Control.Applicative
import Control.Arrow (first, second, (***), (&&&))
import Control.Exception 
import Control.Monad
import Data.Binary
import Data.Foldable(Foldable)
import Data.IORef
import Data.List
import Data.Map (Map)
import Data.Tuple
import Data.Vector.Storable ((!))
import Foreign.C.String
import Foreign.C.Types
import Foreign.ForeignPtr
import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import Foreign.Marshal.Utils
import Foreign.Ptr
import Foreign.Storable
import System.Directory
import System.IO.Error
import System.IO.Unsafe
import Unsafe.Coerce
import qualified Data.ByteString.Lazy as B
import qualified Data.Foldable as F
import qualified Data.Map as Map
import qualified Data.Vector.Unboxed as UV
import qualified Data.Vector as GV
import qualified Data.Vector.Storable as V
import qualified Foreign.Concurrent as C

-- | Intermediary type for interfacing with libsvm. If you need to repeatedly train with the same training data,
--  consider using this type for the training. It is slightly faster and allocates a bit less
type SVMNodes = V.Vector C'svm_node

-- | Class of things that can be interpreted as training vectors for svm. 
class SVMVector a where
    convert :: a -> SVMNodes

instance SVMVector (V.Vector C'svm_node) where
    convert = id

instance SVMVector (V.Vector Double) where
    convert = convertDense 

instance SVMVector (GV.Vector Double) where
    convert = convertDense . GV.convert

instance SVMVector (UV.Vector Double) where
    convert = convertDense . UV.convert

instance SVMVector [Double] where
    convert = convertDense . V.fromList

instance SVMVector (Double,Double) where
    convert (a,b) = convertDense .  V.fromList $ [a,b]

instance SVMVector (Double,Double,Double) where
    convert (a,b,c) = convertDense . V.fromList $ [a,b,c]

instance SVMVector (Double,Double,Double,Double) where
    convert (a,b,c,d) = convertDense . V.fromList $ [a,b,c,d]

instance SVMVector (Double,Double,Double,Double,Double) where
    convert (a,b,c,d,e) = convertDense . V.fromList $ [a,b,c,d,e]

convertDense :: V.Vector Double -> V.Vector C'svm_node
convertDense v = V.generate (dim+1) readVal
    where
        dim = V.length v
        readVal !n | n >= dim = C'svm_node (-1) 0
        readVal !n = C'svm_node (fromIntegral n+1) (double2CDouble $ v ! n)

{-#INLINE double2CDouble #-}
double2CDouble :: Double -> CDouble
double2CDouble = unsafeCoerce

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
        lengths = map (V.length . snd) v
        offsetPtrs addr = take dim 
                          [addr `plusPtr` (idx * sizeOf (C'svm_node undefined undefined)) 
                          | idx <- scanl (+) 0 lengths]
        y   = map (double2CDouble . fst)  v
        xs  = concatMap (V.toList . snd) v

deleteProblem (C'svm_problem l class_array offset_array , node_array) =
    free class_array >> free offset_array >> free node_array 


-- | A Support Vector Machine
newtype SVM = SVM  (ForeignPtr C'svm_model)

getModelPtr (SVM fp) = fp

-- | Somewhat unsafe binary instance. This goes through the disk.
instance Binary SVM where
    put s = put $ idiotToStr s
    get   = get >>= return . idiotFromStr

idiotToStr svm = unsafePerformIO $ withTmp $ \tmp -> do
                    saveSVM tmp svm
                    B.readFile tmp
idiotFromStr svm = unsafePerformIO $ withTmp $ \tmp -> do
                    B.writeFile tmp svm
                    loadSVM tmp
                


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
{-#SPECIALIZE predict :: SVM -> SVMNodes -> Double #-}
predict :: (SVMVector a) => SVM -> a -> Double
predict (getModelPtr -> fptr) 
        (convert -> nodes) = unsafePerformIO $
                           withForeignPtr fptr $ \modelPtr -> 
                           realToFrac <$> V.unsafeWith nodes 
                                             (c'svm_predict modelPtr)

defaultParameters = C'svm_parameter {
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

instance Binary CInt where
    put cint = put (fromIntegral cint :: Int)
    get = fromIntegral <$> (get :: Get Int) 

instance Binary CDouble where
    put cint = put (realToFrac cint :: Double)
    get = realToFrac <$> (get :: Get Double) 

--encodeParams C'svm_parameter{..} = do
--    put c'svm_parameter'svm_type 
--    put c'svm_parameter'kernel_type 
--    put c'svm_parameter'degree 
--    put c'svm_parameter'gamma  
--    put c'svm_parameter'coef0  
--    put c'svm_parameter'cache_size 
--    put c'svm_parameter'eps 
--    put c'svm_parameter'C   
--    put c'svm_parameter'nr_weight 
--    --put c'svm_parameter'weight_label = nullPtr
--    --put c'svm_parameter'weight       = nullPtr
--    put c'svm_parameter'nu 
--    put c'svm_parameter'p  
--    put c'svm_parameter'shrinking
--    put c'svm_parameter'probability 

--decodeParams b = do     
--    c'svm_parameter'svm_type <- get
--    c'svm_parameter'kernel_type <- get
--    c'svm_parameter'degree      <- get   
--    c'svm_parameter'gamma       <- get 
--    c'svm_parameter'coef0       <- get
--    c'svm_parameter'cache_size  <- get 
--    c'svm_parameter'eps         <- get
--    c'svm_parameter'C           <- get
--    c'svm_parameter'nr_weight   <- get
--    c'svm_parameter'nu          <- get
--    c'svm_parameter'p           <- get
--    c'svm_parameter'shrinking   <- get
--    c'svm_parameter'probability <- get
--    let c'svm_parameter'weight_label = nullPtr
--        c'svm_parameter'weight = nullPtr
--    return C'svm_parameter{..}

                   
     

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


withParameters svm kernel ws op = do
    ptr_parameters <- malloc 
    weight_labels  <- newArray (map (fromIntegral.fst) ws)
    weights        <- newArray (map (double2CDouble.snd) ws)
    let no_weights = fromIntegral $ length ws
    poke ptr_parameters parameters{c'svm_parameter'weight_label=weight_labels,c'svm_parameter'weight=weights,c'svm_parameter'nr_weight=no_weights}
    r <- op ptr_parameters
    free ptr_parameters 
    free weights
    free weight_labels
    return r
   where 
        parameters = setTypeParameters svm 
                     . setKernelParameters kernel 
                     $ defaultParameters

-- Other params that currently cannot be passed:
-- epsilon -- termination 0.001
-- cachesize -- in mb 100
-- shrinking -- bool 1
-- probability-estimates -- bool 0
-- weights --

foreign import ccall "wrapper"
  wrapPrintF :: (CString -> IO ()) -> IO (FunPtr (CString -> IO ()))

{-#RULES
  "listToList" F.toList = id #-}

-- |Create an SVM from the training data
{-# SPECIALIZE trainSVM :: (SVMVector a) => SVMType -> Kernel -> [(Int,Double)] -> [(Double, a)] -> IO (String, SVM) #-}
{-# SPECIALIZE trainSVM :: (SVMVector a) => SVMType -> Kernel -> GV.Vector (Int,Double) -> GV.Vector (Double, a) -> IO (String, SVM) #-}
trainSVM :: (Foldable f, SVMVector a) => SVMType -> Kernel -> f (Int,Double) -> f (Double, a) -> IO (String, SVM)
trainSVM svm kernel (F.toList -> ws) (map (second convert) . F.toList -> dataSet) = do
    messages <- newIORef []
    let append x = modifyIORef messages (x:)
    pf <- wrapPrintF (peekCString >=> append) 

    c'svm_set_print_string_function pf
    (problem, ptr_nodes) <- createProblem dataSet
    withParameters svm kernel ws $ \ptr_parameters -> do
        modelPtr <- with problem $ \ptr_problem -> 
                      c'svm_train ptr_problem ptr_parameters
        message  <- unlines . reverse <$> readIORef messages 
        (message ,) . SVM  <$> C.newForeignPtr modelPtr 
                        (deleteProblem (problem, ptr_nodes)
                         >>modelFinalizer modelPtr) 

-- |Cross validate SVM. This is faster than training and predicting for each fold
--  separately, since there are no extra conversions done between libsvm and haskell.
{-#SPECIALIZE crossvalidate :: SVMType -> Kernel -> Int -> [(Double,SVMNodes)] -> IO (String,[Double]) #-}
{-#SPECIALIZE crossvalidate :: SVMType -> Kernel -> Int -> GV.Vector (Double,SVMNodes) -> IO (String,[Double]) #-}
crossvalidate
  :: (Foldable f, SVMVector b) => SVMType -> Kernel -> Int -> f (Double, b) -> IO (String, [Double])
crossvalidate svm kernel folds (map (second convert) . F.toList -> dataSet) = do
    messages <- newIORef []
    let append x = return ()-- modifyIORef messages (x:)
    pf <- wrapPrintF (peekCString >=> append) 
    c'svm_set_print_string_function pf
    (problem, ptr_nodes) <- createProblem dataSet
    withParameters svm kernel [] $ \ptr_parameters-> do
        result_ptr :: Ptr CDouble <- mallocArray (length dataSet)

        with problem $ \ptr_problem -> 
             c'svm_cross_validation ptr_problem ptr_parameters (fromIntegral folds) result_ptr  

        res <- peekArray (length dataSet) result_ptr
        message  <- unlines . reverse <$> readIORef messages 

        free result_ptr >> deleteProblem (problem,ptr_nodes)

        return (message,map realToFrac res)



