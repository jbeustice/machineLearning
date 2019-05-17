/**
 * Logistic regression with k-fold cross-validation using Spark (Scala)
 */

import org.apache.spark.sql.SparkSession

// see less warnings
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// start Session
val spark = SparkSession.builder().getOrCreate()

// load West Nile data
val wn_data = (spark.read.option("header","true")
                       .option("inferSchema","true")
                       .csv("westnile_data.csv"))

// see data
wn_data.printSchema()
wn_data.head(1)

// create categorical variables
import org.apache.spark.ml.feature.{StringIndexer,OneHotEncoderEstimator}

val transform_data = (new StringIndexer()
                    .setInputCol("RESULT")
                    .setOutputCol("resultIndex")
                    .fit(wn_data)
                    .transform(wn_data))

val ready_dataAll = transform_data.select(transform_data("resultIndex").as("label"),
                                      $"TRAP_TYPE",$"SPECIES",$"WEEK",$"NUMBER OF MOSQUITOES")

val ready_data = ready_dataAll.na.drop()

val trapIndexer = new StringIndexer().setInputCol("TRAP_TYPE").setOutputCol("traptypeIndex")
val speciesIndexer = new StringIndexer().setInputCol("SPECIES").setOutputCol("speciesIndex")

val encoder = (new OneHotEncoderEstimator()
              .setInputCols(Array("traptypeIndex","speciesIndex"))
              .setOutputCols(Array("traptypeVec","speciesVec")))

// joins multiple feature columns into a single column of an array of feature values
// (label,features)
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

// dependent var must be titled as "label"; the independent vars as "features"
val assemble = (new VectorAssembler()
               .setInputCols(Array("traptypeIndex","speciesIndex","NUMBER OF MOSQUITOES"))
               .setOutputCol("features"))

// split data
val Array(training, test) = ready_data.randomSplit(Array(0.75, 0.25))

// run k-fold cv for logistic regression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.ml.Pipeline

val lr = new LogisticRegression().setMaxIter(10)

val paramGrid = new ParamGridBuilder().addGrid(lr.regParam,Array(0.1, 0.01)).build()

// cv requires an Estimator, a set of Estimator ParamMaps, and an Evaluator
// 5-fold cv
val cv = (new CrossValidator()
         .setEstimator(lr)
         .setEvaluator(new BinaryClassificationEvaluator)
         .setEstimatorParamMaps(paramGrid)
         .setNumFolds(5))

val pipeline = new Pipeline().setStages(Array(trapIndexer,speciesIndexer,encoder,assemble,cv))

// run cv and choose the best set of parameters.
val cvModel = pipeline.fit(training)

// evaluation --> need to convert to RDD (from df)
import org.apache.spark.mllib.evaluation.MulticlassMetrics

val predictionAndLabels = cvModel.transform(test).select($"prediction",$"label").as[(Double, Double)].rdd

val outcome = new MulticlassMetrics(predictionAndLabels)

// confusion matrix
println("Confusion matrix:")
println(outcome.confusionMatrix)
