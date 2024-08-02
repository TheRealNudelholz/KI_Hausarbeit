package network;

public class ParameterException extends Throwable {

    private final String msg;

    public ParameterException(String msg) {
        this.msg = msg;
    }

    @Override
    public String getMessage() {
        return msg;
    }

    @Override
    public String toString() {
        return msg;
    }

}
