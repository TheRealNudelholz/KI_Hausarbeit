package testing;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import network.ParameterException;

public class Main {

    static ArrayList<Experiment> experiments = new ArrayList<>();
    static double[] pruningRatios = new double[]{0.05, 0.1, 0.2, 0.3, 0.4, 0.5};

    public static void main(String[] args) {
        try {
            experiments.add(new Experiment("MNIST", 10, 50, Experiment.pruning.NONE, 0));
            experiments.add(new Experiment("Linear Function", 100, 1000, Experiment.pruning.NONE, 0));

            for (double pruningRatio : pruningRatios) {

                experiments.add(new Experiment("MNIST", 10, 50, Experiment.pruning.ACTIVATIONLEVEL, pruningRatio));
                experiments.add(new Experiment("MNIST", 10, 50, Experiment.pruning.WEIGHTSTABILITY, pruningRatio));
                experiments.add(new Experiment("MNIST", 10, 50, Experiment.pruning.WEIGHTIMPACT, pruningRatio));
                experiments.add(new Experiment("Linear Function", 10, 10000, Experiment.pruning.ACTIVATIONLEVEL, pruningRatio));
                experiments.add(new Experiment("Linear Function", 10, 10000, Experiment.pruning.WEIGHTSTABILITY, pruningRatio));
                experiments.add(new Experiment("Linear Function", 10, 10000, Experiment.pruning.WEIGHTIMPACT, pruningRatio));

            }

        } catch (ParameterException e) {
            System.err.print(e.getMessage());
        }

        runTests();

    }

    private static void runTests() {
        for (Experiment experiment : experiments) {
            experiment.start();
        }
        for (Experiment experiment : experiments) {
            try {
                experiment.join();
            } catch (InterruptedException e) {
                System.err.print(e.getMessage());
            }
        }

        System.out.println(experiments.get(0));

        for (Experiment experiment : experiments) {
            try {
                PrintWriter out = new PrintWriter(experiment.getTestId() + ".csv");
                out.print(experiment);
                out.close();
            } catch (FileNotFoundException e) {
                System.err.print(e.getMessage());
            }
        }
    }

}
