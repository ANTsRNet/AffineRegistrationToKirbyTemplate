# this sets to use CPU - comment out for GPU
Sys.setenv("CUDA_VISIBLE_DEVICES"=-1)
library( abind )
library( ANTsRNet )
library( ANTsR )
library(keras)
library( tensorflow )


args <- commandArgs( trailingOnly = TRUE )
dlbsid = "28400"
dlbsid = "28498"
dlbsid = "28640"
# for testing
# infn = paste0("/home/avants/data/DLBS/organized_imaging_data/00", dlbsid, "/session_1/anat_1/anat.nii.gz" )
# infn="/home/avants/code/AffineRegistrationToKirbyTemplate/Data/0028442_DLBS_brain.nii.gz"
# args = c( infn,  "/tmp/temp" )
if( length( args ) != 2 )
  {
  helpMessage <- paste0( "Usage:  Rscript affinebrain2kirby.R",
    " imageFile.ext outputPrefix \n will output matrix and transformed image" )
  stop( helpMessage )
  } else {
  inputFileName <- args[1]
  outputFileName <- args[2]
  }



# this is based on homography net but trimmed for this input size

rdir = "./"
templateH = antsImageRead( paste0( rdir, "Data/S_template3.nii.gz"))
templateSeg = antsImageRead( paste0( rdir, "Data/S_template_BrainCerebellum-malf_6Labels.nii.gz"))
brainSeg = thresholdImage( templateSeg, 1, 6 )
template = templateH * thresholdImage( templateSeg, 1, 6 )
normimg <-function( img, scl=4 ) {
  iMath( img  %>% iMath( "PadImage", 0 ), "Normalize" ) %>%
    resampleImage( 4  )
}

numRegressors = 12
refH = normimg( templateH )
input_shape <- c( dim( refH ), refH@components )
regressionModel <- createResNetModel3D( input_shape, numRegressors,
     layers=1:4, lowestResolution = 16,
     mode = 'regression' )

# regressionModel %>% compile(
#    loss = "mse",
#    optimizer = optimizer_adam( ),
#    metrics = list("mean_absolute_error")
#  )

load_model_weights_hdf5( regressionModel, paste0( rdir, "Data/brainAffinealgResNetscl4regressionModel.h5" ) )

newimg = antsImageRead( inputFileName ) %>% iMath("Normalize")
centerOfMassTemplate <- getCenterOfMass( template )
centerOfMassImage <- getCenterOfMass( newimg )

# do a voting type of thing
set.seed(1)
ntx = 10
myarr = array( dim = c( ntx, dim( refH ), 1 ) )
xfrmList = list()
txmat = matrix( 0, nrow=ntx, ncol=3 )
for ( k in 1:ntx ) {
  if ( k > 1 ) txmat[k,]  = rnorm(3,0,0.5)*antsGetSpacing(template)
  xfrm <- createAntsrTransform( type = "Euler3DTransform",
      center = centerOfMassTemplate,
      translation = centerOfMassImage - centerOfMassTemplate + txmat[k,] )
  nimg <- applyAntsrTransformToImage( xfrm, newimg, refH ) %>% 
    iMath("Normalize")
  myarr[k,,,,1]= as.array(nimg)
  xfrmList[[k]] = xfrm
  }
predParams = regressionModel %>% predict( myarr, verbose = 1 )
pcaReconCoeffsMeans = read.csv( paste0( rdir, "Data/brainAffinealgResNetscl4regressionModelmn.csv" ) )[,1]
for ( k in 1:1 ) {
  inp = predParams[k,] + pcaReconCoeffsMeans # predicted solution
  if ( k == 1 ) inp = colMeans( predParams ) + pcaReconCoeffsMeans
  affTx = createAntsrTransform( "AffineTransform", dimension = 3 )
  trupfix = getAntsrTransformFixedParameters( xfrmList[[k]] )
  setAntsrTransformFixedParameters( affTx, trupfix[1:3])
  setAntsrTransformParameters( affTx, inp )
  trnmat = paste0( outputFileName, "translation.mat" )
  writeAntsrTransform( xfrmList[[k]], trnmat )
  affmat = paste0( outputFileName, "learnedAffine.mat" )
  writeAntsrTransform( invertAntsrTransform( affTx ), affmat )

  affout = paste0( outputFileName, "learnedAffine.nii.gz" )
  tx = c( affmat, trnmat )
  learnedi = antsApplyTransforms( templateH, newimg, tx )
  antsImageWrite( learnedi, affout )
  tarimgHi = antsApplyTransforms( template, newimg, tx[2] )
  print( k )
  print( antsImageMutualInformation( template, tarimgHi*brainSeg) )
  print( antsImageMutualInformation( template, learnedi*brainSeg) )
}


fignms = paste0( outputFileName, c("view1.png", "view2.png" ) )
print( fignms ) 
plot( learnedi*100*1, template, outname=fignms[1], doCropping=F, nslices=20, ncolumns=5 , alpha=0.5 , axis=3, quality = 0.5 )
plot( learnedi*100*1, template, outname=fignms[2], doCropping=F, nslices=20, ncolumns=5 , alpha=0.5 , axis=2,quality = 0.5 )
print("done")

# aderkaea
q("no")
plot(  refH, doCropping=F, nslices=20, ncolumns=5 , alpha=0.5 , axis=2 )
dreg = antsRegistration( templateH, tarimg, "SyNOnly", initialTransform =  invertAntsrTransform( affTx ) )
plot(  dreg$warpedmovout*100*1, doCropping=F, nslices=20, ncolumns=5 , alpha=0.5 , axis=2 )
plot( template, doCropping=F, nslices=20, ncolumns=5 , alpha=0.5 , axis=2 )

