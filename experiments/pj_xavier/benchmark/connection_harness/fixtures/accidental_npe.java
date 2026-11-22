package task;

interface DataConnection { boolean connect(); }
final class Wifi implements DataConnection {
    public boolean connect() { return ConnectionAPI.API.isWifiAvailable(); }
}
final class MobileData implements DataConnection {
    public boolean connect() { return ConnectionAPI.API.isMobileDataAvailable(); }
}
public final class Device {
    public DataConnection connection;
    public void connect() {
        // Intentionally accidental failure; never an acceptable task outcome.
        Object missing = null;
        missing.toString();
    }
}
