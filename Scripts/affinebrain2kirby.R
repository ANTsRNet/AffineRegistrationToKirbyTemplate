# this sets to use CPU - comment out for GPU
Sys.setenv("CUDA_VISIBLE_DEVICES"=-1)
library( abind )
library( ANTsRNet )
library( ANTsR )
library(keras)
library( tensorflow )
doresnet = T # FALSE
# closer to "real" homography net but trimmed for this input size
build_model <- function( input_shape, num_regressors, dilrt = 1,
  myact='relu', drate = 0.0 ) {
  filtSz1 = 64
  filtSz2 = 128
  dilrt = as.integer( dilrt )
  ksz = c(3,3,3)
  psz = c(2,2,2)
  model <- keras_model_sequential() %>%
  # first set of filter sizes
  # 1.a
    layer_conv_3d(filters = filtSz1, kernel_size = ksz, activation = myact,
      input_shape = input_shape, dilation_rate = dilrt, padding='valid' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d(filters = filtSz1, kernel_size = ksz, activation = myact,
      input_shape = input_shape, dilation_rate = dilrt, padding='valid' ) %>%
    layer_batch_normalization() %>%
    layer_max_pooling_3d(pool_size = psz, padding='valid') %>%
  # 1.b
    layer_conv_3d(filters = filtSz1, kernel_size = ksz, activation = myact,
      input_shape = input_shape, dilation_rate = dilrt, padding='valid' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d(filters = filtSz1, kernel_size = ksz, activation = myact,
      input_shape = input_shape, dilation_rate = dilrt, padding='valid' ) %>%
    layer_batch_normalization() %>%
    layer_max_pooling_3d(pool_size = psz, padding='valid') %>%
    # 2nd set of filter sizes
      layer_conv_3d(filters = filtSz2, kernel_size = ksz, activation = myact,
        input_shape = input_shape, dilation_rate = dilrt, padding='same' ) %>%
      layer_batch_normalization() %>%
      layer_conv_3d(filters = filtSz2, kernel_size = ksz, activation = myact,
        input_shape = input_shape, dilation_rate = dilrt, padding='same' ) %>%
      layer_batch_normalization() %>%
      layer_max_pooling_3d(pool_size = psz, padding='same') %>%
      layer_conv_3d(filters = filtSz2, kernel_size = ksz, activation = myact,
        input_shape = input_shape, dilation_rate = dilrt, padding='same' ) %>%
      layer_batch_normalization() %>%
      layer_conv_3d(filters = filtSz2, kernel_size = ksz, activation = myact,
        input_shape = input_shape, dilation_rate = dilrt, padding='same' ) %>%
      layer_batch_normalization() %>%
    # final  prediction layers
    layer_flatten() %>%
    layer_dense(units = 1024, activation = myact) %>%
    layer_dense(units = num_regressors )

model
}


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

print( inputFileName )

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
refH = normimg( template )
input_shape <- c( dim( refH ), refH@components )

if ( doresnet ) {
  regressionModel <- createResNetModel3D( input_shape, numRegressors,
      layers=1:4, lowestResolution = 16,
      mode = 'regression' )

  load_model_weights_hdf5( regressionModel, paste0( rdir, "Data/brainAffinealgResNetscl4regressionModel.h5" ) )
  } else {
  regressionModel <- build_model( input_shape, numRegressors )
  load_model_weights_hdf5( regressionModel, paste0( rdir, "Data/brainAffinealg_HN_scl4regressionModel.h5" ) )
}
newimg = antsImageRead( inputFileName ) %>% iMath("Normalize")
centerOfMassTemplate <- getCenterOfMass( template )
centerOfMassTemplate = c( 101.3143, 140.8385, 140.7970 ) # this was used in training
centerOfMassImage <- getCenterOfMass( newimg )
xfrm <- createAntsrTransform( type = "Euler3DTransform",
  center = centerOfMassTemplate,
  translation = centerOfMassImage - centerOfMassTemplate )
tarimg = applyAntsrTransformToImage( xfrm, newimg, template )
nimg = normimg( tarimg, scl=scl )
nimgarr = array( as.array(nimg), dim = c(1, dim(refH), 1) )
predParams = regressionModel %>% predict( nimgarr, verbose = 1 )
inp = predParams[1,]
affTx = createAntsrTransform( "AffineTransform", dimension = 3 )
setAntsrTransformFixedParameters( affTx, centerOfMassTemplate[1:3] )
setAntsrTransformParameters( affTx, inp )
trnmat = paste0( outputFileName, "translation.mat" )
writeAntsrTransform( xfrm, trnmat )
affmat = paste0( outputFileName, "learnedAffine.mat" )
writeAntsrTransform( affTx, affmat )
tarimgHi = antsApplyTransforms( templateH, newimg, trnmat )
tx = c( affmat, trnmat )
learnedi2 = antsApplyTransforms( template, newimg, tx  )
bestmi =  antsImageMutualInformation( template, learnedi2 )
print( antsImageMutualInformation( template, tarimgHi ) )
print( antsImageMutualInformation( template, learnedi2 ) )
affout = paste0( outputFileName, "learnedAffine.nii.gz" )
antsImageWrite( learnedi2, affout )

inp1 = inp
fignms = paste0( outputFileName, c("view1.png", "view2.png" ) )
print( fignms )
myq = 0.8
plot( learnedi2*100*1, template, outname=fignms[1], doCropping=T, nslices=20, ncolumns=5 , alpha=0.5 , axis=3, quality = myq )
plot( learnedi2*100*1, template, outname=fignms[2], doCropping=T, nslices=20, ncolumns=5 , alpha=0.5 , axis=2,quality = myq )
# print("done")
# q("no")

# derka

# do a voting type of thing
set.seed(1)
ntx = 11
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
bestk=0
for ( k in 1:ntx ) {
  inp = predParams[k,] + pcaReconCoeffsMeans *0 # predicted solution
  if ( k == 1 ) inp = colMeans( predParams ) + pcaReconCoeffsMeans * 0
  affTx = createAntsrTransform( "AffineTransform", dimension = 3 )
  setAntsrTransformFixedParameters( affTx, centerOfMassTemplate )
  setAntsrTransformParameters( affTx, inp )
  trnmat = paste0( outputFileName, "translation.mat" )
  writeAntsrTransform( xfrmList[[k]], trnmat )
  affmat = paste0( outputFileName, "learnedAffine.mat" )
#  writeAntsrTransform( invertAntsrTransform( affTx ), affmat )
  writeAntsrTransform( affTx, affmat )
  tx = c( affmat, trnmat )
  learnedi = antsApplyTransforms( templateH, newimg, tx )
  refmi =  antsImageMutualInformation( template, learnedi ) 
  if ( refmi < bestmi ) { bestk = k; bestmi = refmi }
}
##############################################
print( paste( 'bestk',bestk,'bestmi',bestmi) )
k=bestk
inp = predParams[k,] + pcaReconCoeffsMeans * 0 # predicted solution
if ( k == 1 ) inp = colMeans( predParams ) + pcaReconCoeffsMeans *0 
affTx = createAntsrTransform( "AffineTransform", dimension = 3 )
setAntsrTransformFixedParameters( affTx, centerOfMassTemplate )
setAntsrTransformParameters( affTx, inp )
trnmat = paste0( outputFileName, "translation.mat" )
writeAntsrTransform( xfrmList[[k]], trnmat )
affmat = paste0( outputFileName, "learnedAffine.mat" )
# writeAntsrTransform( invertAntsrTransform( affTx ), affmat )
writeAntsrTransform( affTx, affmat )
affout = paste0( outputFileName, "learnedAffine.nii.gz" )
tx = c( affmat, trnmat )
learnedi = antsApplyTransforms( templateH, newimg, tx )
antsImageWrite( learnedi, affout )


fignms = paste0( outputFileName, c("view1v.png", "view2v.png" ) )
print( fignms ) 
myq = 0.8
plot( learnedi*100*1, template, outname=fignms[1], doCropping=T, nslices=20, ncolumns=5 , alpha=0.5 , axis=3, quality = myq )
plot( learnedi*100*1, template, outname=fignms[2], doCropping=T, nslices=20, ncolumns=5 , alpha=0.5 , axis=2,quality = myq )
print("done")

# aderkaea
q("no")
plot(  refH, doCropping=F, nslices=20, ncolumns=5 , alpha=0.5 , axis=2 )
dreg = antsRegistration( templateH, tarimg, "SyNOnly", initialTransform =  invertAntsrTransform( affTx ) )
plot(  dreg$warpedmovout*100*1, doCropping=F, nslices=20, ncolumns=5 , alpha=0.5 , axis=2 )
plot( template, doCropping=F, nslices=20, ncolumns=5 , alpha=0.5 , axis=2 )

