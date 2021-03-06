package wineClassification;

import java.io.UnsupportedEncodingException;
import java.net.URLDecoder;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

import scala.Tuple2;

public class LogisticRegressionPrediction {

	public static void main(String[] args) {
	
		SparkConf conf = new SparkConf().setAppName("WineClassification").setMaster("local");
		
		conf.set("spark.testing.memory", "2147480000");
		
	    JavaSparkContext jsc = new JavaSparkContext(conf);

	    String test_path = args[0];
		JavaRDD<String> test_data = jsc.textFile(test_path);
		
		String first_t = test_data.first();
	    JavaRDD<String> test_filtered_data = test_data.filter((String s) -> {return !s.contains(first_t);});
	
	    
		JavaRDD<LabeledPoint> test_parsed_data = test_filtered_data.map(line -> {
		  String[] parts = line.split(";");
		  double[] points = new double[parts.length - 1];
	        for (int i = 0; i < (parts.length - 1); i++) {
	            points[i] = Double.valueOf(parts[i]);
	        }
	        return new LabeledPoint(Double.valueOf(parts[parts.length - 1]), Vectors.dense(points));
	    });
	
		test_parsed_data.cache();

		LogisticRegressionModel model = LogisticRegressionModel.load(jsc.sc(),
				args[1]);

		JavaPairRDD<Object, Object> predictionAndLabels = test_parsed_data.mapToPair(p ->
		  new Tuple2<>(model.predict(p.features()), p.label()));

		MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
		double accuracy = metrics.accuracy();
		double f_score = metrics.weightedFMeasure();
		
		System.out.println();
		System.out.println("----------------------------------------------------------------------------");
		System.out.println("Accuracy = " + accuracy);
		System.out.println("----------------------------------------------------------------------------");
		System.out.println();
		
		System.out.println();
		System.out.println("----------------------------------------------------------------------------");
		System.out.println("F Measure = " + f_score);
		System.out.println("----------------------------------------------------------------------------");
		System.out.println();
	    jsc.stop();

	}

}
