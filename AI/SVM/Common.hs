module AI.SVM.Common where

import Data.Char
import System.Random.MWC
import Control.Applicative
import Control.Monad
import System.Directory
import Control.Exception

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
        

