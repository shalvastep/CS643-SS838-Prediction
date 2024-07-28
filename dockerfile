# Use the official Ubuntu image as the base image
FROM ubuntu:latest

# Install Java (required for Spark)
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk wget tar

# Install Spark
RUN wget https://downloads.apache.org/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz && \
    tar xzf spark-3.5.1-bin-hadoop3.tgz && \
    mv spark-3.5.1-bin-hadoop3 /opt/spark && \
    rm spark-3.5.1-bin-hadoop3.tgz

# Set environment variables for Spark
ENV SPARK_HOME /opt/spark
ENV PATH $SPARK_HOME/bin:$PATH

# Copy your application JAR file into the container
COPY prediction-1.0-SNAPSHOT-jar-with-dependencies.jar /opt/spark/jars/

# Set the entry point to run your Spark application
ENTRYPOINT ["spark-submit", "--class", "com.cs643.project2.Main", "--master", "local[1]", "/opt/spark/jars/prediction-1.0-SNAPSHOT-jar-with-dependencies.jar"]