package network;

import java.io.Serializable;
import java.util.Random;

public class Connection implements Serializable {

    public Neuron sourceNode;
    public Neuron targetNode;

    private double weight;

    private double previousWeight;
    private double sumOfDeltas = 0;
    private int numberOfWeightUpdates = 0;

    public Connection(Neuron source, Neuron target) {
        this.sourceNode = source;
        this.targetNode = target;
        weight = new Random().nextDouble() * 0.5;
    }

    public Double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        double difference = Math.abs(previousWeight - weight);
        previousWeight = this.weight;
        this.weight = weight;
        sumOfDeltas += difference;
        numberOfWeightUpdates++;
    }

    public Double getInstability() {
        return sumOfDeltas / numberOfWeightUpdates;
    }

}
