/**
 * PCA (after standardization) using Spark (Scala)
 */

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.PCA

// see less warnings
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// start Session
val spark = SparkSession.builder().getOrCreate()

// load FIFA data
val fifa_data = (spark.read.option("header","true")
                        .option("inferSchema","true")
                        .csv("fifa_data.csv"))

// see data
fifa_data.printSchema()
fifa_data.head(1)

// joins multiple feature columns into a single column of an array of feature values
import org.apache.spark.ml.feature.{VectorAssembler,StandardScaler}
import org.apache.spark.ml.linalg.Vectors

val dfALL = (fifa_data.select($"Crossing",$"Finishing",$"HeadingAccuracy",$"ShortPassing",
$"Volleys",$"Dribbling",$"Curve",$"FKAccuracy",$"LongPassing",$"BallControl",$"Acceleration",
$"SprintSpeed",$"Agility",$"Reactions",$"Balance",$"ShotPower",$"Jumping",$"Stamina",
$"Strength",$"LongShots",$"Aggression",$"Interceptions",$"Positioning",$"Vision",$"Penalties",
$"Composure",$"Marking",$"StandingTackle",$"SlidingTackle",$"GKDiving",$"GKHandling",
$"GKKicking",$"GKPositioning",$"GKReflexes"))

val df = dfALL.na.drop

val colnames = (Array("Crossing","Finishing","HeadingAccuracy","ShortPassing",
"Volleys","Dribbling","Curve","FKAccuracy","LongPassing","BallControl","Acceleration",
"SprintSpeed","Agility","Reactions","Balance","ShotPower","Jumping","Stamina",
"Strength","LongShots","Aggression","Interceptions","Positioning","Vision","Penalties",
"Composure","Marking","StandingTackle","SlidingTackle","GKDiving","GKHandling",
"GKKicking","GKPositioning","GKReflexes"))

val assemble = (new VectorAssembler().setInputCols(colnames)
                                     .setOutputCol("unscaledFeatures"))

// Use the assembler to transform df to column of unscaled features
val ready_data = assemble.transform(df).select($"unscaledFeatures")

// standardize data
val scaler = (new StandardScaler()
  .setInputCol("unscaledFeatures")
  .setOutputCol("features")
  .setWithStd(true)
  .setWithMean(false))

val scalerFit = scaler.fit(ready_data)

val scaled_data = scalerFit.transform(ready_data)

// run PCA with 10 PCs
val pcaALL = (new PCA()
  .setInputCol("features")
  .setOutputCol("pcaFeatures")
  .setK(10)
  .fit(scaled_data))

// view explained variance to choose optimal # of PCs
val exVar = pcaALL.explainedVariance

// run PCA with optimal # of PCs (4)
val pca = (new PCA()
  .setInputCol("features")
  .setOutputCol("pcaFeatures")
  .setK(4)
  .fit(scaled_data))

val exVarOptimal = pca.explainedVariance

val pcaDF = pca.transform(scaled_data)

// view pcaFeatures
val result = pcaDF.select("pcaFeatures")
result.head(1)
