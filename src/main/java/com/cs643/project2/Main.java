package com.cs643.project2;


import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

public class Main {

    public static void main(String[] args) {

        System.out.println("Starting in Prediction application");

        String validationDataFilePath = args[0];

        System.out.println(validationDataFilePath);

        Properties props = new Properties();

        try (InputStream input = Main.class.getClassLoader().getResourceAsStream("config.properties")) {
            if (input == null) {
                throw new FileNotFoundException("config.properties not found in classpath");
            }
            props.load(input);
        } catch (IOException e) {
            e.printStackTrace();
        }


        // Spark session
        SparkSession spark = SparkSession.builder()
                .appName("QualityPredictionApplication")
                .master("local")
                .config("spark.hadoop.fs.s3a.access.key", props.getProperty("spark.hadoop.fs.s3a.access.key"))
                .config("spark.hadoop.fs.s3a.secret.key", props.getProperty("spark.hadoop.fs.s3a.secret.key"))
                .config("spark.hadoop.fs.s3a.endpoint", props.getProperty("spark.hadoop.fs.s3a.endpoint"))
                .config("spark.hadoop.fs.s3a.path.style.access", props.getProperty("spark.hadoop.fs.s3a.path.style.access"))
                .getOrCreate();


        LogisticRegressionModel model = LogisticRegressionModel.load(props.getProperty("s3.model.src"));


        // loading validation data
        Dataset<Row> testData = spark.read()
                .format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .option("delimiter", ";")
                .load(validationDataFilePath);


        testData.printSchema(5);

        String[] columns = {
                "fixed acidity",
                "volatile acidity",
                "citric acid",
                "residual sugar",
                "chlorides",
                "free sulfur dioxide",
                "total sulfur dioxide",
                "density",
                "pH",
                "sulphates",
                "alcohol",
                "quality"
        };

        // Create a VectorAssembler to combine feature columns into a single vector column
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(columns)
                .setOutputCol("features");


        // Tata includes the feature vector
        Dataset<Row> testDataWithFeatures = assembler.transform(testData);

        // Predicting
        Dataset<Row> predictions = model.transform(testDataWithFeatures);

        System.out.println("show 5 rows");
        predictions.show(20);

        // Show only the "features" column
        predictions.select("features").show(10);

        System.out.println("logged features");
        predictions.select("prediction", "features").show();


        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");

        double f1Score = evaluator.evaluate(predictions);

        System.out.println("*******************************");
        System.out.println("F1 Score = " + f1Score);
        System.out.println("*******************************");

        // Stop Spark session
        System.out.println("Application ");
        spark.stop();

    }
}
