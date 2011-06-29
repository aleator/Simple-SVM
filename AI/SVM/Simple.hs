{-# LANGUAGE ForeignFunctionInterface, BangPatterns, ScopedTypeVariables, TupleSections, 
             RecordWildCards #-}
-------------------------------------------------------------------------------
-- |
-- Module     : Bindings.SVM
-- Copyright  : (c) 2011 Paulo Tanimoto, Ville Tirronen
-- License    : BSD3
--
-- Maintainer : Ville Tirronen <aleator@gmail.com>
--              Paulo Tanimoto <ptanimoto@gmail.com>
--
-------------------------------------------------------------------------------
-- For a high-level description of the C API, refer to the README file 
-- included in the libsvm archive, available for download at 
-- <http://www.csie.ntu.edu.tw/~cjlin/libsvm/>.

module AI.SVM.Simple (loadSVM, saveSVM
                 ,trainSVM, predict
                 , SVM
                 , SVMType(..), Kernel(..))  where

import qualified Data.Vector.Storable as V
import Data.Vector.Storable ((!))
import Bindings.SVM
import Foreign.C.Types
import Foreign.C.String
import Foreign.Ptr
import Foreign.ForeignPtr
import qualified Foreign.Concurrent as C
import Foreign.Marshal.Utils
import Control.Applicative
import System.IO.Unsafe
import Foreign.Storable
import Control.Monad


{-# SPECIALIZE convertDense :: V.Vector Double -> V.Vector C'svm_node #-}
{-# SPECIALIZE convertDense :: V.Vector Float -> V.Vector C'svm_node #-}
convertDense :: (V.Storable a, Real a) => V.Vector a -> V.Vector C'svm_node
convertDense v = V.generate (dim+1) readVal
    where
        dim = V.length v
        readVal !n | n >= dim = C'svm_node (-1) 0
        readVal !n = C'svm_node (fromIntegral n) (realToFrac $ v ! n)


withProblem :: [(Double, V.Vector Double)] -> (Ptr C'svm_problem -> IO b) -> IO b
withProblem v op = -- Well. This turned out super ugly. Also, this is a veritable
                   -- bug magnet.
                   V.unsafeWith xs $ \ptr_xs ->
                   V.unsafeWith y  $ \ptr_y -> 
                    let optrs = offsetPtrs ptr_xs
                    in V.unsafeWith optrs $ \ptr_offsets ->
                        with (C'svm_problem (fromIntegral dim) ptr_y ptr_offsets) op
    where 
        dim = length v
        lengths = map (V.length . snd) v
        offsetPtrs addr = V.fromList . take dim $
                          [addr `plusPtr` (idx * sizeOf (xs ! 0))
                          | idx <- scanl (+) 0 lengths]
        y   = V.fromList . map (realToFrac . fst) $ v
        xs  = V.concat . map (extractSvmNode.snd) $ v
        extractSvmNode x = convertDense $ V.generate (V.length x) (x !)

-- | A Support Vector Machine
newtype SVM = SVM (ForeignPtr C'svm_model)


modelFinalizer :: Ptr C'svm_model -> IO ()
modelFinalizer modelPtr = with modelPtr c'svm_free_and_destroy_model

-- | load an svm from a file.
loadSVM :: FilePath -> IO SVM
loadSVM fp = do
    ptr <- withCString fp c'svm_load_model
    let fin = modelFinalizer ptr
    SVM <$> C.newForeignPtr ptr fin

-- | Save an svm to a file.
saveSVM :: FilePath -> SVM -> IO ()
saveSVM fp (SVM fptr) = 
    withForeignPtr fptr $ \model_ptr -> 
    withCString fp      $ \cstr      ->
    c'svm_save_model cstr model_ptr


-- | Predict the class of a vector with an SVM.
predict :: SVM -> V.Vector Double -> Double
predict (SVM fptr) vec = unsafePerformIO $
                           withForeignPtr fptr $ \modelPtr -> 
                           let nodes = convertDense vec
                           in realToFrac <$> V.unsafeWith nodes (c'svm_predict modelPtr)

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

-- | SVM variants
data SVMType = 
               -- | C svm (the default tool for classification tasks)
               C_SVC  {cost :: Double}
               -- | Nu svm
             | NU_SVC {cost :: Double, nu :: Double}
               -- | One class svm
             | ONE_CLASS {nu :: Double}
               -- | Epsilon support vector regressor
             | EPSILON_SVR {cost :: Double, epsilon :: Double}
               -- | Nu support vector regressor 
             | NU_SVR {cost :: Double, nu :: Double}

-- | SVM kernel type
data Kernel = Linear 
            | Polynomial {gamma :: Double, coef0 :: Double, degree :: Int}
            | RBF {gamma :: Double}
            | Sigmoid {gamma :: Double, coef0 :: Double}
            deriving (Show)

rf = realToFrac
setKernelParameters Linear p = p
setKernelParameters (Polynomial {..}) p = p{c'svm_parameter'gamma=rf gamma
                                           ,c'svm_parameter'coef0=rf coef0
                                           ,c'svm_parameter'degree=fromIntegral degree}
setKernelParameters (RBF {..}) p        = p{c'svm_parameter'gamma=rf gamma }
setKernelParameters (Sigmoid {..}) p    = p{c'svm_parameter'gamma=rf gamma
                                           ,c'svm_parameter'coef0=rf coef0 }

setTypeParameters (C_SVC cost) p     = p{c'svm_parameter'C=rf cost}

setTypeParameters (NU_SVC{..}) p     = p{c'svm_parameter'C=rf cost
                                        ,c'svm_parameter'nu=rf nu}
setTypeParameters (ONE_CLASS{..}) p  = p{c'svm_parameter'nu=rf nu}

setTypeParameters (EPSILON_SVR{..}) p = p{c'svm_parameter'C=rf cost
                                        ,c'svm_parameter'p=rf epsilon}

setTypeParameters (NU_SVR {..}) p    = p{c'svm_parameter'C=rf cost
                                        ,c'svm_parameter'nu=rf nu}


withParameters svm kernel op = with parameters op
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

-- | Create an SVM from the training data
trainSVM :: SVMType -> Kernel -> [(Double, V.Vector Double)] -> IO SVM
trainSVM svm kernel dataSet = do
    pf <- wrapPrintF (\cstr -> peekCString cstr >>= print . (, ":HS")) 
          -- The above is just a test. Realistically at that point there
          -- should be an ioref that captures the output which would then
          -- be returned from this function.
    c'svm_set_print_string_function pf
    modelPtr <- withProblem dataSet $ \ptr_problem ->
                withParameters svm kernel $ \ptr_parameters ->
                c'svm_train ptr_problem ptr_parameters
    SVM <$> C.newForeignPtr modelPtr (modelFinalizer modelPtr) 




