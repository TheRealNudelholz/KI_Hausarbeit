package network;

import java.io.Serializable;
import java.util.ArrayList;

public class Neuron implements Serializable {

    public final ArrayList<Connection> outgoing;
    public final ArrayList<Connection> incoming;

    private double activation = 0;
    private double activationSum = 0;
    private double delta;
    private double bias;
    private double output;
    private double error;
    private final Activation activationFunction;

    private double inactiveTransfers;

    public Neuron(Activation activationFunction, int initialBias) {
        outgoing = new ArrayList<>();
        incoming = new ArrayList<>();
        bias = initialBias;
        this.activationFunction = activationFunction;
    }

    public void calcDelta() {
        this.delta = this.error * this.getDerivative();
    }

    public void calcBias(double learningRate) {
        this.bias -= learningRate * this.delta;
    }

    public double calcError(double expected) {
        this.error = this.output - expected;
        return this.error;
    }

    public double getDerivative() {
        return this.activationFunction.derivative(activation);
    }

    public void transfer() {
        output = activationFunction.phi(activation);

        if (this.activationFunction instanceof Relu) {
            if (output <= 0.01) {
                inactiveTransfers++;
            }
        } else {
            activationSum++;
        }
    }

    public void resetError() {
        this.error = 0;
    }

    public void addToError(double value) {
        this.error += value;
    }

    public double getError() {
        return this.error;
    }

    public Double getActivationSum() {
        return this.activationSum;
    }

    public void setActivation(double activation) {
        this.activation = activation;
    }

    public double getOutput() {
        return this.output;
    }

    public void setOutput(double output) {
        this.output = output;
    }

    public double getDelta() {
        return this.delta;
    }

    public Double getInactiveTransfers() {
        return this.inactiveTransfers;
    }

    public double getBias() {
        return this.bias;
    }

}
