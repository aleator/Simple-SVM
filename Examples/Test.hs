import AI.SVM.Simple

main = do
    let trnC = [(x==y,[x,y]) | x <- [0,1], y <- [0,1::Double]]
        trnR = [(x-y,[x,y])  | x <- [0,0.1..1], y <- [0,0.1..1::Double]]
        (msgs1,cf) = trainClassifier (C 0.5) (RBF 0.3) trnC
        (msgs3,cf2) = trainWtdClassifier (C 0.5) (RBF 0.3) [(True,10),(False,1)] trnC
        (msgs2,re) = trainRegressor  (Epsilon 0.1 0.1) (RBF 0.3) trnR
    print msgs1
    print msgs2
    print msgs3
    print (map (classify cf) $ map snd trnC) 
    print (map (classify cf2) $ map snd trnC) 
    print (map (predictRegression re) $ [[0,1],[0.5,0.2],[1,2::Double]]) 
