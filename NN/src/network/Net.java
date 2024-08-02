package network;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

public class Net implements Serializable {

    int[] dimension;
    Layer inputLayer;
    Layer outputLayer;
    ArrayList<Layer> hiddenLayers;

    ArrayList<Connection> connections;

    double[][] input;
    double[][] expected;
    ArrayList<Double> outputErrors;

    final Activation defaultActivation = new Sigmoid();
    Activation hiddenLayerActivation;
    Activation outputLayerActivation;
    final int defaultBias = -1;

    Pruner pruner;

    public boolean useWeightImpactPruning;
    public boolean useWeightStabilityPruning;
    public boolean useActivationPruning;

    public Net(int[] layerDimensions, Activation hiddenLayerActivation, Activation outputLayerActivation) {
        this.hiddenLayerActivation = hiddenLayerActivation;
        this.outputLayerActivation = outputLayerActivation;
        this.dimension = layerDimensions;

        init();
    }

    public Net(int[] layerDimensions, Activation hiddenLayerActivation, Activation outputLayerActivation, Pruner pruner) {
        this(layerDimensions, hiddenLayerActivation, outputLayerActivation);
        this.pruner = pruner;
    }

    private void init() {
        outputErrors = new ArrayList<>();
        inputLayer = new Layer();
        hiddenLayers = new ArrayList<>();
        outputLayer = new Layer();
        connections = new ArrayList<>();
        for (int i = 0; i < dimension[0]; i++) {
            inputLayer.nodes.add(new Neuron(defaultActivation, defaultBias));
        }
        for (int i = 1; i < dimension.length - 1; i++) {
            Layer hiddenLayer = new Layer();
            for (int j = 0; j < dimension[i]; j++) {
                hiddenLayer.nodes.add(new Neuron(this.hiddenLayerActivation, defaultBias));
            }
            hiddenLayers.add(hiddenLayer);
        }
        for (int i = 0; i < dimension[dimension.length - 1]; i++) {
            outputLayer.nodes.add(new Neuron(this.outputLayerActivation, defaultBias));
        }
        addConnections();
    }

    public void reset() {
        this.init();
    }

    public double[] predict(double[] input) {
        for (int i = 0; i < inputLayer.nodes.size(); i++) {
            inputLayer.nodes.get(i).setOutput(input[i]);
        }
        for (Layer hiddenLayer : hiddenLayers) {
            hiddenLayer.calcInput();
        }
        outputLayer.calcInput();
        return outputLayer.getOutput();
    }

    public double measureErrorOnTestData(double[][] testInputs, double[][] testTargets) {
        double error = 0;
        for (int sample = 0; sample < testInputs.length; sample++) {
            double[] predicted = predict(testInputs[sample]);
            error += getAvgOutputError(predicted, testTargets[sample]);
        }
        return error / testInputs.length;
    }

    public double getAvgOutputError(double output[], double[] target) {
        double totalError = 0;
        for (int i = 0; i < output.length; i++) {
            totalError += Math.abs(output[i] - target[i]);
        }
        return totalError / output.length;
    }

    public void shuffleData() {
        Random rnd = new Random();
        for (int i = 0; i < input.length; i++) {
            int randomIndexToSwap = rnd.nextInt(input.length);
            double[] temp = input[randomIndexToSwap];
            input[randomIndexToSwap] = input[i];
            input[i] = temp;
            temp = expected[randomIndexToSwap];
            expected[randomIndexToSwap] = expected[i];
            expected[i] = temp;
        }

    }

    public double training(double[][] input, double[][] expected, double learningRate, int repetitions,
            boolean shuffle) {
        this.input = input;
        this.expected = expected;

        double accuracy = 0;
        for (int r = 0; r < repetitions; r++) {
            if (shuffle) {
                shuffleData();
            }
            accuracy = 0;
            for (int sample = 0; sample < this.input.length; sample++) {
                // Forward pass
                double[] output = predict(this.input[sample]);

                double meanError = 0;
                // Calculate error for output layer
                for (int node = 0; node < outputLayer.nodes.size(); node++) {
                    Neuron outputNode = outputLayer.nodes.get(node);
                    // Error of output node is difference between its output and the expected one
                    meanError += Math.abs(outputNode.calcError(this.expected[sample][node]));

                    // The weights delta is the product of the nodes error and the partial derivative of its output
                    outputNode.calcDelta();
                }
                outputErrors.add(meanError);

                // Backpropagate the error to hidden layers
                for (int layer = hiddenLayers.size() - 1; layer >= 0; layer--) {
                    Layer hiddenLayer = hiddenLayers.get(layer);
                    for (Neuron hiddenNode : hiddenLayer.nodes) {
                        hiddenNode.resetError();
                        for (Connection edge : hiddenNode.outgoing) {
                            // The error of the hidden node sums up with the weights and errors of the connected nodes
                            hiddenNode.addToError(edge.getWeight() * edge.targetNode.getError());

                        }
                        // Its weight delta is calculated similar to the output nodes
                        hiddenNode.calcDelta();
                    }
                }

                // Weights have to update after backpropagation
                updateWeights(learningRate, outputLayer);
                for (Layer hiddenLayer : hiddenLayers) {
                    updateWeights(learningRate, hiddenLayer);
                }
                // Calculate mean accuracy
                accuracy = 0;
                for (int j = 0; j < output.length; j++) {
                    accuracy += Math.abs(this.expected[sample][j] - output[j]);
                }
                accuracy /= output.length;
            }
            prune();

        }
        return 1 - accuracy;
    }

    public void prune() {
        if (useActivationPruning) {
            this.hiddenLayers = pruner.activationPruning(hiddenLayers, connections, hiddenLayerActivation);
            this.connections = pruner.prunedConnections;
        }
        if (useWeightImpactPruning) {
            this.connections = pruner.weightImpactPruning(hiddenLayers, connections);
        }
        if (useWeightStabilityPruning) {
            this.connections = pruner.weightStabilityPruning(hiddenLayers, connections);
        }
    }

    private void updateWeights(double learningRate, Layer layer) {
        for (Neuron node : layer.nodes) {
            for (Connection edge : node.incoming) {
                double currentWeight = edge.getWeight();
                edge.setWeight(currentWeight - learningRate * node.getDelta() * edge.sourceNode.getOutput());
            }
            node.calcBias(learningRate);
        }
    }

    // Adds connections between every node of a layer with every node of the following layer
    private void addConnections() {
        for (Neuron inputNode : inputLayer.nodes) {
            for (Neuron hiddenNode : hiddenLayers.get(0).nodes) {
                Connection connection = new Connection(inputNode, hiddenNode);
                inputNode.outgoing.add(connection);
                hiddenNode.incoming.add(connection);
                connections.add(connection);
            }
        }
        for (int i = 1; i < hiddenLayers.size(); i++) {
            for (Neuron hiddenNode : hiddenLayers.get(i).nodes) {
                for (Neuron previousNode : hiddenLayers.get(i - 1).nodes) {
                    Connection connection = new Connection(previousNode, hiddenNode);
                    previousNode.outgoing.add(connection);
                    hiddenNode.incoming.add(connection);
                    connections.add(connection);
                }
            }
        }
        for (Neuron outputNode : outputLayer.nodes) {
            for (Neuron hiddenNode : hiddenLayers.get(hiddenLayers.size() - 1).nodes) {
                Connection connection = new Connection(hiddenNode, outputNode);
                hiddenNode.outgoing.add(connection);
                outputNode.incoming.add(connection);
                connections.add(connection);
            }
        }
    }

    public int getNumberOfConnections() {
        return connections.size();
    }
}
