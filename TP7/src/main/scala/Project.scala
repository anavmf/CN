import java.util.logging.{Level, Logger}

import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.reflect.io.Path
import scala.util.Try

/**
  * Created by fernando on 10-03-2017.
  */
object MLlibProject {

  var log: Logger = _
  var spark: SparkContext = _
  val numFeatures : Int = 3000
  val step : Double = 0.05
  val c : Double = 0.1
  val numIterations : Int = 100

  /**
    * ARGUMENTS:
    * (0) - mode: train or classify
    *       train: train a model using the given loadDatasetPath, then save it to disk
    *       classify: load the model from disk and classify another loadDatasetPath
    * (1) - loadDatasetPath: the loadDatasetPath to use, either on "train" or "classify" mode
    * (2) - modelPath: The path to save/load the model
    */
  def main(args: Array[String]): Unit = {
    if(args.size != 3){
      println("usage: simple-project_2.11-1.0.jar <mode> <loadDatasetPath> <modelPath>")
      println("where")
      println("\tmode is \"train\" or \"classify\"")
      println("\tloadDatasetPath is the path to the dataset to be used")
      println("\tmodelPath is the ABSOLUTE path where to save/load the model")
      System.exit(-1)
    }
  
    log = Logger.getLogger(getClass.getName)

    spark = new SparkContext(new SparkConf().setAppName("ClassProject"))
    log.log(Level.INFO, "Spark session created; beginning program execution")

    val mode = args(0)
    val dataset = args(1)
    val modelPath = args(2)
    log.log(Level.INFO, "arg0: " + mode)
    log.log(Level.INFO, "arg1: " + dataset)
    log.log(Level.INFO, "arg2: " + modelPath)

    mode match {
      case "train" => trainModel(dataset, modelPath)
      case "classify" => classify(dataset, modelPath)
    }

    log.log(Level.INFO, "Execution finished; exiting")
    spark.stop()
  }

  /**
    * Trains a model and stores it to disk
    * @param datasetPath The path to the dataset used to train the model
    * @param saveModelPath The path where to save the model
    */
  private def trainModel(datasetPath : String, saveModelPath: String) : Unit = {
    val (_, datasetWithFeatures) = getFeaturesSet(datasetPath)
    log.log(Level.INFO, "Beginning model training")
    val svm = SVMWithSGD.train(datasetWithFeatures, numIterations = numIterations, stepSize = step, regParam = c)
    svm.clearThreshold()
    log.log(Level.INFO, "Model trained")
    val path: Path = Path (saveModelPath + "SVMModel")
    if(path.exists) {
      log.log(Level.INFO, "Model exists; overwriting")
      Try(path.deleteRecursively())
    }
    /* when using spark context to access local files (i.e., not not HDFS or Mesos) it is required to use "file://"
       or the execution simply stops; actually, spark is waiting for the file to become available on disk, which never
       will because spark is trying to read the file from the wrong place (HDFS by default if I am not mistaken) */
    svm.save(spark, "file://" + saveModelPath + "SVMModel")
    log.log(Level.INFO, "Model saved  to " + saveModelPath)
  }

  /**
    * Classifies the tweets as security relevant or not
    * It loads an already trained model from disk, then reads a dataset and classifies it
    * @param loadDatasetPath the path to where the dataset to be classified is stored
    * @param loadModelPath the path to where the model is stored
    */
  private def classify(loadDatasetPath: String, loadModelPath: String) : Unit = {
    val (tweetsRDD, datasetWithFeatures) = getFeaturesSet(loadDatasetPath)

    val svm = SVMModel.load(spark, "file://" + loadModelPath + "SVMModel")

    //calculate what the model predicts for each feature vector
    //keep the predictions paired with the correct label for further processing simplicity
    val predictions = datasetWithFeatures.map(point => (point.label, svm.predict(point.features)))
    new ModelMetrics(predictions.collect).printResults(log)

    //what this line does is to produce on RDD[label, prediction, features, tweet] and then keep only the features and
    //tweets of the tweets classified as positive (i.e., security relevant)
    val positiveTweets = predictions.map(_._2).zip(datasetWithFeatures.map(_.features)).zip(tweetsRDD)
      .map(t => (t._1._1, t._1._2, t._2)).filter(_._1 > 0.0).map(t => (t._2, t._3))

    cluster(positiveTweets)
  }

  /**
    * Reads a dataset from file and returns the corresponding labelled point set
    * The dataset is composed of lines in the following format:
    *   "<label>\t<pre-processed_tweet>"
    * the label is 0 (not security relevant) or 1 (security relevant)
    * A labelled point is a data format composed of the label and the feature vector of the tweet
    * @param dataPath The path where the tweet dataset is stored
    * @return A set of labelled points
    */
  private def getFeaturesSet(dataPath : String) : (RDD[String], RDD[LabeledPoint]) = {
    //read lines from file; no RDD here as splitting the lines and separating labels from tweets would require
    //creating more unnecessary RDDs
    val lines = Source.fromFile(dataPath).getLines()
    val (labels, tweets) = lines.map(_.split("\t")).map(a => (a(0).toInt, a(1))).toList.unzip
    //now that we have them separated we can create RDDs
    //the tweets are converted to features
    //the labels are added later to the features to create an RDD of labelled points
    val labelsRDD = spark.parallelize(labels)
    val tweetsRDD = spark.parallelize(tweets)

    log.log(Level.INFO, "Tweet loadDatasetPath loaded")
    val features = tweetsToFeatures(tweetsRDD)
    log.log(Level.INFO, "Tweets converted to features")
    (tweetsRDD, labelsRDD.zip(features).map(pair => new LabeledPoint(pair._1, pair._2)))
  }

  /**
    * Converts the Tweets to feature vectors
    * @param data An RDD containing the Tweets
    * @return A corresponding feature (Vector) RDD
    */
  private def tweetsToFeatures(data: RDD[String]) : RDD[Vector] = {
    val hashingTF = new HashingTF(numFeatures)
    //hashingTF iterates over whole words of the sentence
    val tf = data.map(s => hashingTF.transform(s.split(" ")))

    // While applying HashingTF only needs a single pass to the data, applying IDF needs two passes:
    // First to compute the IDF vector and second to scale the term frequencies by IDF.
    tf.cache()
    val idf = new IDF().fit(tf)
    idf.transform(tf)
  }

  /**
    * Perform k-means clustering of the positive tweets in order to simplify the presentation of the positive tweets
    * @param rdd An RDD containing the positive tweets and their corresponding feature vectors
    */
  private def cluster(rdd : RDD[(Vector, String)]) : Unit = {
    var numClusters = 1
    var kModel: KMeansModel = null
    var SSE: Double = 0
    var currentSSE: Double = Double.MaxValue
    val features = rdd.map(_._1)
    //keep the tweets on a map for easy access when printing the clusters
    val map : Map[Vector, String] = rdd.collect().toMap[Vector, String]

    do {
      kModel = KMeans.train(features, numClusters, numIterations)
      SSE = kModel.computeCost(features)

      //SSE is the sum of squared errors of the model; basically, it measures the quality of the model for 'k' clusters
      //we increase the number of clusters and gather the SSE
      //while it decreases, a model with k+1 clusters is better than a model with k clusters
      if (SSE > 0 && SSE <= currentSSE) {
        currentSSE = SSE
        numClusters += 1
      }
    } while (SSE == currentSSE)

    val clusters : ArrayBuffer[ArrayBuffer[Vector]] = ArrayBuffer.fill[Vector](numClusters, 0)(null)
    val clusterList :Seq[Vector] = features.collect()

    //collect the tweets and distribute them to their clusters
    for(v <- clusterList)
      clusters(kModel.predict(v)) :+= v.asInstanceOf[Vector]

    var i = 1
    for (cluster <- clusters) {
      println("Cluster " + i)
      for (f <- cluster) {
        println("\t" + map(f.toSparse))
      }
      i += 1
    }
  }

  class ModelMetrics(labelsAndScore: Seq[(Double, Double)]) {
    val positive = 1.0
    val negative = 0.0

    var truePositive, falsePositive, trueNegative, falseNegative, positiveLabel, negativeLabel, truePositiveRate,
    trueNegativeRate, accuracy, precision, recall, tpTNAverage, fMeasure, label, prediction: Double = 0

    for (s <- labelsAndScore) {
      label = s._1.asInstanceOf[Double]
      prediction = if (s._2.asInstanceOf[Double] > 0.0) 1.0 else 0.0

      if (label == positive && prediction == positive)
        truePositive += 1
      else if (prediction == positive && label == negative)
        falseNegative += 1
      else if (prediction == negative && label == negative)
        trueNegative += 1
      else //score == negative && label == positive
        falsePositive += 1

      if (label == 1.0)
        positiveLabel += 1
      else
        negativeLabel += 1
    }

    val numberOfDecimals : Int = 5

    //these are confirmed
    if(positiveLabel > 0)
      truePositiveRate = BigDecimal(truePositive / positiveLabel).setScale(numberOfDecimals, BigDecimal.RoundingMode.HALF_UP).toDouble
    if(negativeLabel > 0)
      trueNegativeRate = BigDecimal(trueNegative / negativeLabel).setScale(numberOfDecimals, BigDecimal.RoundingMode.HALF_UP).toDouble

    if(positiveLabel + negativeLabel > 0)
      accuracy = BigDecimal((truePositive + trueNegative) / (positiveLabel + negativeLabel)).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
    if(truePositive + falsePositive > 0)
      precision = BigDecimal(truePositive / (truePositive + falsePositive)).setScale(numberOfDecimals, BigDecimal.RoundingMode.HALF_UP).toDouble
    if(truePositive + falseNegative > 0)
      recall = BigDecimal(truePositive / (truePositive + falseNegative)).setScale(numberOfDecimals, BigDecimal.RoundingMode.HALF_UP).toDouble

    tpTNAverage = BigDecimal((truePositiveRate + trueNegativeRate) / 2).setScale(numberOfDecimals, BigDecimal.RoundingMode.HALF_UP).toDouble
    if(precision + truePositiveRate > 0)
      fMeasure = BigDecimal(2 * (precision * truePositiveRate) / (precision + truePositiveRate)).setScale(numberOfDecimals, BigDecimal.RoundingMode.HALF_UP).toDouble

    def printResults(log : Logger): Unit = {
      log.log(Level.INFO, "True positive rate:\t" + truePositiveRate)
      log.log(Level.INFO, "True negative rate:\t" + trueNegativeRate)
      log.log(Level.INFO, "Average TP/TN:\t" + tpTNAverage + "\n")
      log.log(Level.INFO, "Accuracy:\t\t" + accuracy)
      log.log(Level.INFO, "Precision:\t\t" + precision)
      log.log(Level.INFO, "Recall:\t\t" + recall+"\n")
      log.log(Level.INFO, "F-measure:\t\t" + fMeasure)
    }
  }
}
