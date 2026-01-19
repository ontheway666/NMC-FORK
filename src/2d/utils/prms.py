bOutputBCWeight = 0
bVorIsCoord= 0

bCoordTranslation = 0
translationY = 0.7
translationX = 1.5


bFitField = 0
biggestField=[-1.1,  1.9,  -0.59,  0.6]  +  [translationX, translationX, translationY, translationY]

assert(not(bCoordTranslation and bFitField))


bOutBC = 0

bBasicVel = 0

bEnhanceGradP = 0
bSampleBigger = 0


btangConstraint = 0


assert(not (bSampleBigger and bOutBC))