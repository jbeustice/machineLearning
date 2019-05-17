/**
 * Linear regression using Spark (Scala)
 */

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression

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
// (label,features)
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

// dependent var must be titled as "label"; the independent vars as "features"
val df = (h_data.select(h_data("Price").as("label"),
                       $"Bed",$"Bath",$"SqFt",$"Floors",$"YearBuilt",$"AreaHomeValue",
                       $"AreaPctOwn",$"AreaPctOwn",$"AreaIncome"))

val assemble = (new VectorAssembler().setInputCols(Array("Bed","Bath","SqFt","Floors",
                                                        "YearBuilt","AreaHomeValue",
                                                        "AreaPctOwn","AreaPctVacant",
                                                        "AreaIncome")).setOutputCol("features"))

// Use the assembler to transform df to two columns
val ready_data = assemble.transform(df).select($"label",$"features")

// linear regression
val lr = new LinearRegression()
val lrModel = lr.fit(ready_data)
val lrSummary = lrModel.summary

// show results
df.columns
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
lrSummary.pValues
lrSummary.r2
