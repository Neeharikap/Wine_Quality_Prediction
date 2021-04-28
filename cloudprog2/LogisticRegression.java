package wineClassification;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

import scala.Tuple2;

public class LogisticRegression {

	public static void main(String[] args) {
	// --------------------------------------- Start Spark Context --------------------------------------- //
		// use line below if running on Eclipse 
		//SparkConf conf = new SparkConf().setAppName("WineClassification").setMaster("local");
		
		// use line below if running on a cluster
		SparkConf conf = new SparkConf().setAppName("WineClassification");
		
	    JavaSparkContext jsc = new JavaSparkContext(conf);
	
	    
	// --------------------------------------- Load and Parse the Training Data --------------------------------------- //
		//String train_path = "src/main/java/wineClassification/TrainingDataset.csv";
	    String train_path = "s3n://wine-classification/TrainingDataset.csv";
		JavaRDD<String> train_data = jsc.textFile(train_path);
		
		// filter out header
		String first_t = train_data.first();
	    JavaRDD<String> train_filtered_data = train_data.filter((String s) -> {return !s.contains(first_t);});
	
	    
		JavaRDD<LabeledPoint> train_parsed_data = train_filtered_data.map(line -> {
		  String[] parts = line.split(";");
		  double[] points = new double[parts.length - 1];
	        for (int i = 0; i < (parts.length - 1); i++) {
	            points[i] = Double.valueOf(parts[i]);
	        }
	        return new LabeledPoint(Double.valueOf(parts[parts.length - 1]), Vectors.dense(points));
	    });
		
	
		train_parsed_data.cache();
		
	// --------------------------------------- Load and Parse the Training Data --------------------------------------- //
		//String valid_path = "src/main/java/wineClassification/ValidationDataset.csv";
	    String valid_path = "s3n://wine-classification/ValidationDataset.csv";
		JavaRDD<String> validation_data = jsc.textFile(valid_path);
		
		// filter out header
		String firstV = validation_data.first();
	    JavaRDD<String> valid_filtered_data = validation_data.filter((String s) -> {return !s.contains(firstV);});
	
	    
		JavaRDD<LabeledPoint> valid_parsed_data = valid_filtered_data.map(line -> {
		  String[] parts = line.split(";");
		  double[] points = new double[parts.length - 1];
	        for (int i = 0; i < (parts.length - 1); i++) {
	            points[i] = Double.valueOf(parts[i]);
	        }
	        return new LabeledPoint(Double.valueOf(parts[parts.length - 1]), Vectors.dense(points));
	    });
		
		valid_parsed_data.cache();
	
		
	// --------------------------------------- Build Model --------------------------------------- //
		LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
				.setNumClasses(10)
				.run(train_parsed_data.rdd());
		
		
	// --------------------------------------- Validate Model --------------------------------------- //
		// Compute raw scores on the test set.
		JavaPairRDD<Object, Object> predictionAndLabels = valid_parsed_data.mapToPair(p ->
		  new Tuple2<>(model.predict(p.features()), p.label()));


	// ---------------------------------- Calculate Accuracy and F-Score ---------------------------- //
		MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
		double accuracy = metrics.accuracy();
		System.out.println();
		System.out.println("----------------------------------------------------------------------------");
		System.out.println("Validation Accuracy = " + accuracy);
		System.out.println("----------------------------------------------------------------------------");
		System.out.println();
		
		//System.out.format("Weighted precision = %f\n", metrics.weightedPrecision());
		//System.out.format("Weighted recall = %f\n", metrics.weightedRecall());
		
		double f_score = metrics.weightedFMeasure();
		System.out.println();
		System.out.println("----------------------------------------------------------------------------");
		System.out.println("Validation F Measure = " + f_score);
		System.out.println("----------------------------------------------------------------------------");
		System.out.println();
		
	// ---------------------------------------- Save Model ---------------------------------------- //
		//model.save(jsc.sc(), "src/main/java/wineClassification/LogisticRegressionModel");
		model.save(jsc.sc(), "s3n://wine-classification/LogisticRegressionModel");
		
	// ---------------------------------------- Stop Spark Context ---------------------------------- //
		jsc.stop();

	}

}
