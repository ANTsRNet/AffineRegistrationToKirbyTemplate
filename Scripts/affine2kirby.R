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
# args = c( infn,  "/tmp/temp" )
if( length( args ) != 2 )
  {
  helpMessage <- paste0( "Usage:  Rscript affine2kirby.R",
    " imageFile.ext outputPrefix \n will output matrix and transformed image" )
  stop( helpMessage )
  } else {
  inputFileName <- args[1]
  outputFileName <- args[2]
  }



# this is based on homography net but trimmed for this input size
build_model <- function( input_shape, num_regressors, dilrt = 1,
  myact='relu', drate = 0.0 ) {
  filtSz1 = 64
  filtSz2 = 128
  dilrt = as.integer( dilrt )
  ksz = c(3,3,3)
  psz = c(2,2,2)
  model <- keras_model_sequential() %>%
  # first set of filter sizes
    layer_conv_3d(filters = filtSz1, kernel_size = ksz, activation = myact,
      input_shape = input_shape, dilation_rate = dilrt ) %>%
    layer_batch_normalization() %>%
    layer_max_pooling_3d(pool_size = psz) %>%
    layer_conv_3d(filters = filtSz1, kernel_size = ksz, activation = myact,
      input_shape = input_shape, dilation_rate = dilrt ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d(filters = filtSz1, kernel_size = ksz, activation = myact,
      input_shape = input_shape, dilation_rate = dilrt ) %>%
    layer_batch_normalization() %>%
    layer_max_pooling_3d(pool_size = psz) %>%
    # 2nd set of filter sizes
      layer_conv_3d(filters = filtSz2, kernel_size = ksz, activation = myact,
        input_shape = input_shape, dilation_rate = dilrt ) %>%
      layer_batch_normalization() %>%
      layer_max_pooling_3d(pool_size = psz) %>%
      layer_conv_3d(filters = filtSz2, kernel_size = ksz, activation = myact,
        input_shape = input_shape, dilation_rate = dilrt ) %>%
      layer_batch_normalization() %>%
      layer_max_pooling_3d(pool_size = psz) %>%
    layer_flatten() %>%
    layer_dense(units = 1024, activation = myact) %>%
    layer_dense(units = num_regressors )

model
}

rdir = "./"
templateH = antsImageRead( paste0( rdir, "Data/S_template3.nii.gz"))
templateSeg = antsImageRead( paste0( rdir, "Data/S_template_BrainCerebellum-malf_6Labels.nii.gz"))
template = templateH * thresholdImage( templateSeg, 1, 6 )
normimg <-function( img, scl=4 ) {
  iMath( img  %>% iMath( "PadImage", 0 ), "Normalize" ) %>%
    resampleImage( 4  )
}

numRegressors = 10
refH = normimg( templateH )
input_shape <- c( dim( refH ), refH@components )
regressionModel <- build_model( input_shape, numRegressors )

regressionModel %>% compile(
    loss = "mse",
    optimizer = optimizer_adam( ),
    metrics = list("mean_absolute_error")
  )

load_model_weights_hdf5( regressionModel, paste0( rdir, "Data/regiwDeformationBasisalg_BM_scl4regressionModel.h5" ) )


newimg = antsImageRead( inputFileName )
treg = antsRegistration( templateH, newimg,
  "Translation", regIterations=c(100,0,0,0), outprefix=outputFileName )
tarimg = antsApplyTransforms( templateH, newimg, treg$fwdtransforms )
nimg = normimg( tarimg )
nimgarr = array( as.array(nimg), dim = c(1, dim(refH), 1) )
predParams = regressionModel %>% predict( nimgarr, verbose = 1 )
basisw = data.matrix( read.csv( paste0( rdir, "Data/regiwDeformationBasisalg_BM_scl4regressionModelbasis.csv" ) ))
pcaReconCoeffsMeans = read.csv( paste0( rdir, "Data/regiwDeformationBasisalg_BM_scl4regressionModelmn.csv" ) )[,1]
inp = data.matrix( basisw ) %*% t(matrix( predParams + pcaReconCoeffsMeans, nrow=1 ))# predicted solution
affTx = createAntsrTransform( "AffineTransform", dimension = 3 )
trupfix = getAntsrTransformFixedParameters( readAntsrTransform( treg$fwd[1] ) )
setAntsrTransformFixedParameters( affTx, trupfix )
setAntsrTransformParameters( affTx, inp )
affmat = paste0( outputFileName, "learnedAffine.mat" )
writeAntsrTransform( invertAntsrTransform( affTx ), affmat )

affout = paste0( outputFileName, "learnedAffine.nii.gz" )
tx = c( affmat, treg$fwdtransforms )
learnedi = antsApplyTransforms( templateH, tarimg, tx[1] )
antsImageWrite( learnedi, affout )

print( antsImageMutualInformation( template, tarimg ) )
print( antsImageMutualInformation( template, learnedi ) )
fignms = paste0( outputFileName, c("view1.png", "view2.png" ) )
plot( learnedi*100*1, template, outname=fignms[1], doCropping=F, nslices=20, ncolumns=5 , alpha=0.5 , axis=3, quality = 0.5 )
plot( learnedi*100*1, template, outname=fignms[2], doCropping=F, nslices=20, ncolumns=5 , alpha=0.5 , axis=2,quality = 0.5 )

q("no")
plot(  refH, doCropping=F, nslices=20, ncolumns=5 , alpha=0.5 , axis=2 )
dreg = antsRegistration( templateH, tarimg, "SyNOnly", initialTransform =  invertAntsrTransform( affTx ) )
plot(  dreg$warpedmovout*100*1, doCropping=F, nslices=20, ncolumns=5 , alpha=0.5 , axis=2 )
plot( template, doCropping=F, nslices=20, ncolumns=5 , alpha=0.5 , axis=2 )

