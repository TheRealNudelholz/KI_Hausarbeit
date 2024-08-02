package network;

import java.io.Serializable;

public abstract class Activation implements Serializable {

    public abstract double phi(double x);

    public abstract double derivative(double x);

}
