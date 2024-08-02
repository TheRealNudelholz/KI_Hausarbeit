package testing;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.time.LocalDateTime;
import network.Net;
import network.ParameterException;
import network.Pruner;
import network.Relu;
import network.Sigmoid;

public class Experiment extends Thread {

    private Net network;

    private double[][] trainingDurations;
    private double[][] predictionDurations;
    private double[][] outputErrorsOnIteration;
    private int[][] numbersOfConnections;
    private int repetitions;
    private String dataset;

    private double[] avgPredictionDurations;
    private double[] avgTrainingDurations;
    private double[] avgErrorsOnIteration;
    private int[] avgNumbersOfConnections;

    private double[][] trainingInput;
    private double[][] trainingTarget;

    private double[][] testInput;
    private double[][] testTarget;

    private Pruner pruner;
    private double pruningRatio;
    private pruning pruningMethod;

    private int[] layerDimensions;
    private double learningRate;
    private boolean shuffleTrainingData;
    private int trainingIterations;
    private int epochsPerTrainingIteration;
    private int sampleSize;

    private boolean useWeightImpactPruning;
    private boolean useWeightStabilityPruning;
    private boolean useActivationPruning;

    public Experiment(String dataset, int repetitions, int trainingIterations, pruning pruningMethod, double pruningRatio) throws ParameterException {
        this.dataset = dataset;
        this.repetitions = repetitions;
        this.trainingIterations = trainingIterations;
        this.pruningRatio = pruningRatio;
        this.pruningMethod = pruningMethod;

        this.trainingDurations = new double[repetitions][trainingIterations];
        this.predictionDurations = new double[repetitions][trainingIterations];
        this.outputErrorsOnIteration = new double[repetitions][trainingIterations];
        this.numbersOfConnections = new int[repetitions][trainingIterations];

        this.pruner = new Pruner();
        pruner.setRatio(this.pruningRatio);

        switch (pruningMethod) {
            case pruning.WEIGHTIMPACT:
                this.useWeightImpactPruning = true;
                break;
            case pruning.WEIGHTSTABILITY:
                this.useWeightStabilityPruning = true;
                break;
            case pruning.ACTIVATIONLEVEL:
                this.useActivationPruning = true;
                break;
            default:
                this.useWeightImpactPruning = false;
                this.useWeightStabilityPruning = false;
                this.useActivationPruning = false;
                break;
        }

        switch (dataset) {
            case "MNIST" -> {
                this.learningRate = 0.01;
                this.sampleSize = 30000;
                this.epochsPerTrainingIteration = 1;
                this.shuffleTrainingData = false;
                this.layerDimensions = new int[]{784, 50, 50, 10};
                this.network = new Net(layerDimensions, new Relu(), new Sigmoid(), pruner);
                synchronized (this) {
                    this.generateMNIST();
                }
            }
            default -> {
                this.learningRate = 0.05;
                this.sampleSize = 10000;
                this.epochsPerTrainingIteration = 1;
                this.shuffleTrainingData = true;
                this.layerDimensions = new int[]{1, 10, 10, 10, 1};
                this.network = new Net(layerDimensions, new Sigmoid(), new Sigmoid(), pruner);
                this.generateExampleData(sampleSize, "lin");
            }

        }
        // set best found hyperparameter for given dataset

        network.useActivationPruning = useActivationPruning;
        network.useWeightImpactPruning = useWeightImpactPruning;
        network.useWeightStabilityPruning = useWeightStabilityPruning;
    }

    @Override
    public void run() {
        System.out.println("Starting Experiment: " + getTestId());
        // Multiple repetitions for archieving average testresults
        for (int i = 0; i < repetitions; i++) {
            // Multiple training sessions to see the networks training progress
            for (int j = 0; j < trainingIterations; j++) {
                numbersOfConnections[i][j] = network.getNumberOfConnections();
                // Learn the training data
                double startTime = System.currentTimeMillis();
                network.training(trainingInput, trainingTarget, learningRate, epochsPerTrainingIteration, shuffleTrainingData);
                double endTime = System.currentTimeMillis();
                double duration = endTime - startTime;
                trainingDurations[i][j] = duration;

                // Test the network with testdata
                startTime = System.currentTimeMillis();
                double avgError = network.measureErrorOnTestData(testInput, testTarget);
                endTime = System.currentTimeMillis();
                double avgPredictionTime = endTime - startTime;

                outputErrorsOnIteration[i][j] = avgError;
                predictionDurations[i][j] = avgPredictionTime;
            }
            network.reset();
        }
        System.out.println("Finished Experiment: " + getTestId());
    }

    @Override
    public String toString() {
        calcAverageTestResults();
        String result = "Dataset,Training iteration,Pruning method,Pruning ratio,Output error,Duration (Training),Duration (Prediction),Number of connections";
        for (int i = 0; i < trainingIterations; i++) {
            result += "\n" + dataset + "," + i + "," + pruningMethod + "," + pruningRatio + "," + avgErrorsOnIteration[i] + "," + avgTrainingDurations[i] + ","
                    + avgPredictionDurations[i] + "," + avgNumbersOfConnections[i];
        }
        return result;
    }

    private void calcAverageTestResults() {
        this.avgPredictionDurations = new double[trainingIterations];
        this.avgTrainingDurations = new double[trainingIterations];
        this.avgErrorsOnIteration = new double[trainingIterations];
        this.avgNumbersOfConnections = new int[trainingIterations];

        for (int i = 0; i < trainingIterations; i++) {
            double avgPredictionDuration = 0;
            double avgTrainingDuration = 0;
            double avgError = 0;
            double avgNumberOfConnections = 0;

            for (int j = 0; j < repetitions; j++) {
                avgPredictionDuration += predictionDurations[j][i];
                avgTrainingDuration += trainingDurations[j][i];
                avgError += outputErrorsOnIteration[j][i];
                avgNumberOfConnections += numbersOfConnections[j][i];
            }
            avgPredictionDurations[i] = avgPredictionDuration / repetitions;
            avgTrainingDurations[i] = avgTrainingDuration / repetitions;
            avgErrorsOnIteration[i] = avgError / repetitions;
            avgNumbersOfConnections[i] = (int) (avgNumberOfConnections / repetitions);
        }
    }

    private synchronized void generateMNIST() {
        double[][] inputs = new double[this.sampleSize][784];
        double[][] targets = new double[this.sampleSize][10];

        int counter = 0;
        try {
            FileReader filereader = new FileReader("mnist_train.csv");
            try (BufferedReader reader = new BufferedReader(filereader)) {
                String line = reader.readLine();
                while (line != null && counter < this.sampleSize) {
                    String[] eintraege = line.split(",");
                    int z = Integer.parseInt(eintraege[0]);
                    targets[counter][z] = 1;
                    for (int i = 1; i < 785; i++) {
                        inputs[counter][i - 1] = Double.parseDouble(eintraege[i]);
                    }
                    counter++;
                    line = reader.readLine();
                }
            }
        } catch (IOException | NumberFormatException ex) {
            System.err.println("Cannot read training data: " + ex.getMessage());
        }
        // Normalize data
        for (double[] input : inputs) {
            for (int i = 0; i < input.length; i++) {
                input[i] = input[i] / 255;
            }
        }
        this.trainingInput = inputs;
        this.trainingTarget = targets;

        inputs = new double[10000][784];
        targets = new double[10000][10];

        counter = 0;
        try {
            FileReader filereader = new FileReader("mnist_test.csv");
            try (BufferedReader reader = new BufferedReader(filereader)) {
                String line = reader.readLine();
                while (line != null && counter < 10000) {
                    String[] entries = line.split(",");
                    int targetNumber = Integer.parseInt(entries[0]);
                    targets[counter][targetNumber] = 1;
                    for (int i = 1; i < 785; i++) {
                        inputs[counter][i - 1] = Double.parseDouble(entries[i]);
                    }
                    counter++;
                    line = reader.readLine();
                }
            }
        } catch (IOException | NumberFormatException ex) {
            System.err.println("Cannot read training data: " + ex.getMessage());
        }
        this.testInput = inputs;
        this.testTarget = targets;

    }

    private void generateExampleData(int size, String function) {

        double[][] inputs = new double[size][1];
        double[][] targets = new double[size][1];

        switch (function) {
            case "sin" -> {
                for (int i = 0; i < size; i++) {
                    inputs[i][0] = Math.random();
                    targets[i][0] = (Math.sin(inputs[i][0]) + 1) / 2;
                }
            }
            default -> {
                for (int i = 0; i < size; i++) {
                    inputs[i][0] = Math.random();
                    targets[i][0] = inputs[i][0] / 2;
                }
            }
        }
        this.trainingInput = inputs;
        this.trainingTarget = targets;

        inputs = new double[size][1];
        targets = new double[size][1];

        switch (function) {
            case "sin" -> {
                for (int i = 0; i < size; i++) {
                    inputs[i][0] = Math.random();
                    targets[i][0] = (Math.sin(inputs[i][0]) + 1) / 2;
                }
            }
            default -> {
                for (int i = 0; i < size; i++) {
                    inputs[i][0] = Math.random();
                    targets[i][0] = inputs[i][0] / 2;
                }
            }
        }
        this.testInput = inputs;
        this.testTarget = targets;

    }

    public String getTestId() {
        String result = LocalDateTime.now().toString() + "_" + dataset + "_" + pruningMethod.toString().replace(" ", "_") + "_" + ((int) (pruningRatio * 100));
        return result;
    }

    public enum pruning {
        WEIGHTIMPACT("Weight impact"),
        WEIGHTSTABILITY("Weight stability"),
        ACTIVATIONLEVEL("Activation level"),
        NONE("No pruning");
        private final String string;

        pruning(String name) {
            string = name;
        }

        @Override
        public String toString() {
            return string;
        }
    }

}
