package network;

import java.io.Serializable;
import java.util.ArrayList;

public class Layer implements Serializable {

    public ArrayList<Neuron> nodes;

    private double[] output;

    public Layer() {
        nodes = new ArrayList<>();
    }

    public void calcInput() {
        for (Neuron node : nodes) {
            double net = 0;
            for (Connection edge : node.incoming) {
                net += edge.getWeight() * edge.sourceNode.getOutput();
            }
            net += node.getBias();
            node.setActivation(net);
            //node.activation = net;
            node.transfer();
        }
    }

    public double[] getOutput() {
        output = new double[nodes.size()];
        for (int i = 0; i < nodes.size(); i++) {
            output[i] = nodes.get(i).getOutput();
        }
        return output;
    }
}
