package task;

interface DataConnection { boolean connect(); }
final class Wifi implements DataConnection {
    public boolean connect() { return true; }
}
final class MobileData implements DataConnection {
    public boolean connect() { return true; }
}
public final class Device {
    public DataConnection connection;
    public void connect() {
        // Intentionally wrong: ignore available types and actual connection success.
        connection = new Wifi();
    }
}
