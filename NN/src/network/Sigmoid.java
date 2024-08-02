package network;

public class Sigmoid extends Activation {

    private final String name = "Sigmoid";

    @Override
    public double phi(double x) {
        return 1.0 / (1 + Math.exp(-x));
    }

    @Override
    public double derivative(double x) {
        x = phi(x);
        return x * (1 - x);
    }

    @Override
    public String toString() {
        return this.name;
    }
}
