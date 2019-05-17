/**
 * Kmeans clustering (after standardization) using Spark (Scala)
 */

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.clustering.KMeans

// see less warnings
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// start Session
val spark = SparkSession.builder().getOrCreate()

// load housing data
val h_data = (spark.read.option("header","true")
                        .option("inferSchema","true")
                        .csv("housing_data.csv"))

// see data
h_data.printSchema()
h_data.head(1)

// joins multiple feature columns into a single column of an array of feature values
import org.apache.spark.ml.feature.{VectorAssembler,StandardScaler}
import org.apache.spark.ml.linalg.Vectors

val df = h_data.select($"Price",$"AreaHomeValue",$"AreaIncome",$"Lat",$"Long")

val assemble = (new VectorAssembler().setInputCols(Array("Price","AreaHomeValue","AreaIncome","Lat","Long"))
                                     .setOutputCol("unscaledFeatures"))

// Use the assembler to transform df to column of features
val ready_data = assemble.transform(df).select($"unscaledFeatures")

// standardize data
val scaler = (new StandardScaler()
  .setInputCol("unscaledFeatures")
  .setOutputCol("features")
  .setWithStd(true)
  .setWithMean(false))

val scalerFit = scaler.fit(ready_data)

val scaled_data = scalerFit.transform(ready_data)

// subset data to determine optimal value for k
val fit_k = scaled_data.sample(false,0.3)

// fit data via kmeans and return SSE
def eval_k(num:Int): Double = {
  val kmeans = new KMeans().setK(num)
  val model = kmeans.fit(fit_k)
  return model.computeCost(fit_k)
}

// prints k and within set SSE for range of k
var k = 0
for(k <- 2 to 15){
  val wsSSE = eval_k(k)
  println(s"(K = $k, wsSSE = $wsSSE)")
}

// fit full data via kmeans given optimal k
val bestK = 5
val kmeans = new KMeans().setK(bestK)
val model = kmeans.fit(scaled_data)

// results
val wsSSE = model.computeCost(ready_data)
println(s"(K = $bestK, wsSSE = $wsSSE)")

println("Cluster Centers: ")
model.clusterCenters.foreach(println)
