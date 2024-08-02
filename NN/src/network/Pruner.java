package network;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class Pruner implements Serializable {

    private double ratio = 0.2;
    public ArrayList<Connection> prunedConnections;

    public void setRatio(double pruningRatio) throws ParameterException {
        if (pruningRatio >= 1) {
            throw new ParameterException("Ratio cannot be 1 or higher. It would remove all connections of the neural net.");
        }
        ratio = pruningRatio;
    }

    public double getRatio() {
        return ratio;
    }

    public ArrayList<Layer> activationPruning(ArrayList<Layer> hiddenLayers, ArrayList<Connection> connections, Activation activation) {

        if (activation instanceof Relu) {
            for (int j = 0; j < hiddenLayers.size() - 1; j++) {
                Layer hiddenLayer = hiddenLayers.get(j);
                // inactive transfers descending
                hiddenLayer.nodes.sort((a, b) -> b.getInactiveTransfers().compareTo(a.getInactiveTransfers()));
                for (int i = 0; i < hiddenLayer.nodes.size() * getRatio() - 1; i++) {
                    connections.removeAll(hiddenLayer.nodes.get(i).incoming);
                    hiddenLayer.nodes.remove(hiddenLayer.nodes.get(i));
                }
            }
        } else {
            for (int j = 0; j < hiddenLayers.size() - 1; j++) {
                Layer hiddenLayer = hiddenLayers.get(j);
                // activation value ascending
                hiddenLayer.nodes.sort((a, b) -> b.getActivationSum().compareTo(a.getActivationSum()));
                for (int i = 0; i < hiddenLayer.nodes.size() * getRatio() - 1; i++) {
                    connections.removeAll(hiddenLayer.nodes.get(i).incoming);
                    hiddenLayer.nodes.remove(hiddenLayer.nodes.get(i));
                }
            }
        }
        prunedConnections = connections;
        return hiddenLayers;
    }

    public ArrayList<Connection> weightStabilityPruning(List<Layer> hiddenLayers, ArrayList<Connection> connections) {

        ArrayList<Connection> connectionsToBeRemoved = new ArrayList<>();

        for (int i = 0; i < hiddenLayers.size() - 1; i++) {
            Layer hiddenLayer = hiddenLayers.get(i);
            for (Neuron node : hiddenLayer.nodes) {
                // instability descending
                node.outgoing.sort((a, b) -> b.getInstability().compareTo(a.getInstability()));
                connectionsToBeRemoved.addAll(node.outgoing.subList(0, (int) (node.outgoing.size() * getRatio() - 1)));
            }
        }
        connections.removeAll(connectionsToBeRemoved);
        return connections;
    }

    public ArrayList<Connection> weightImpactPruning(List<Layer> hiddenLayers, ArrayList<Connection> connections) {

        ArrayList<Connection> connectionsToBeRemoved = new ArrayList<>();

        for (int i = 0; i < hiddenLayers.size() - 1; i++) {
            Layer hiddenLayer = hiddenLayers.get(i);
            for (Neuron node : hiddenLayer.nodes) {
                // weight ascending
                node.outgoing.sort((a, b) -> a.getWeight().compareTo(a.getWeight()));
                connectionsToBeRemoved.addAll(node.outgoing.subList(0, (int) (node.outgoing.size() * getRatio() - 1)));
            }
        }
        connections.removeAll(connectionsToBeRemoved);
        return connections;
    }
}
