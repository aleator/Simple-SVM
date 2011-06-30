{-# LANGUAGE ForeignFunctionInterface, BangPatterns, ScopedTypeVariables,
             TupleSections, ViewPatterns, RecordWildCards #-}
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
                 ,getNRClasses
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
import Foreign.Marshal.Array
import Foreign.Marshal.Alloc
import Control.Applicative
import System.IO.Unsafe
import Foreign.Storable
import Control.Monad
import System.Directory


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
                          [addr `plusPtr` (idx * sizeOf (head xs)) -- #TODO: Safer alternative to head
                          | idx <- scanl (+) 0 lengths]
        y   = map (realToFrac . fst)  v
        xs  = concatMap (V.toList . extractSvmNode . snd) $ v
        extractSvmNode x = convertDense $ V.generate (V.length x) (x !)

deleteProblem (C'svm_problem l class_array offset_array , node_array) =
    free class_array >> free offset_array >> free node_array 


-- | A Support Vector Machine
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
    unless e $ error "Model does not exist"
        -- Not finding the file causes a bus error. Could do without that..
        -- #TODO: Make a smarter error
    ptr <- withCString fp c'svm_load_model
    let fin = modelFinalizer ptr
    SVM <$> C.newForeignPtr ptr fin

-- | Save an svm to a file.
saveSVM :: FilePath -> SVM -> IO ()
saveSVM fp (getModelPtr -> fptr) = 
    withForeignPtr fptr $ \model_ptr -> 
    withCString fp      $ \cstr      ->
    c'svm_save_model cstr model_ptr

getNRClasses (getModelPtr -> fptr) 
    = fromIntegral <$>  withForeignPtr fptr c'svm_get_nr_class

-- | Predict the class of a vector with an SVM.
predict :: SVM -> V.Vector Double -> Double
predict (getModelPtr -> fptr) vec = unsafePerformIO $
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

setTypeParameters (C_SVC cost) p     = p{c'svm_parameter'C=rf cost
                                        ,c'svm_parameter'svm_type=c'C_SVC}

setTypeParameters (NU_SVC{..}) p     = p{c'svm_parameter'C=rf cost
                                        ,c'svm_parameter'nu=rf nu
                                        ,c'svm_parameter'svm_type=c'NU_SVC}

setTypeParameters (ONE_CLASS{..}) p  = p{c'svm_parameter'nu=rf nu
                                        ,c'svm_parameter'svm_type=c'ONE_CLASS}

setTypeParameters (EPSILON_SVR{..}) p = p{c'svm_parameter'C=rf cost
                                        ,c'svm_parameter'p=rf epsilon
                                        ,c'svm_parameter'svm_type=c'EPSILON_SVR}

setTypeParameters (NU_SVR {..}) p    = p{c'svm_parameter'C=rf cost
                                        ,c'svm_parameter'nu=rf nu
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

-- | Create an SVM from the training data
trainSVM :: SVMType -> Kernel -> [(Double, V.Vector Double)] -> IO SVM
trainSVM svm kernel dataSet = do
    pf <- wrapPrintF (const $ return ()) --(\cstr -> peekCString cstr >>= print . (, ":HS")) 
          -- The above is just a test. Realistically at that point there
          -- should be an ioref that captures the output which would then
          -- be returned from this function.
    c'svm_set_print_string_function pf
    (problem, ptr_nodes) <- createProblem dataSet
    ptr_parameters <- malloc 
    poke ptr_parameters (setParameters svm kernel)
    modelPtr <- with problem $ \ptr_problem -> 
                  c'svm_train ptr_problem ptr_parameters
    SVM <$> C.newForeignPtr modelPtr 
                    (free ptr_parameters>>deleteProblem (problem, ptr_nodes)
                    >>modelFinalizer modelPtr) 




