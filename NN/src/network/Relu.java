package network;

public class Relu extends Activation {

    private final String name = "Relu";

    private final double alpha = 0.01;

    @Override
    public double phi(double x) {
        if (x > 0) {
            return x;
        } else {
            return alpha * x;
        }
    }

    @Override
    public double derivative(double x) {
        if (x > 0) {
            return 1;
        } else {
            return alpha;
        }
    }

    @Override
    public String toString() {
        return this.name;
    }
}
