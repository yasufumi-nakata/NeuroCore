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
        java.util.ArrayList<String> present = DeviceAPI.API.getAvailableConnections();
        if (present.contains("wifi")) {
            Wifi candidate = new Wifi();
            if (candidate.connect()) { connection = candidate; return; }
        }
        if (present.contains("mobiledata")) {
            MobileData candidate = new MobileData();
            if (candidate.connect()) { connection = candidate; return; }
        }
        throw new RuntimeException("No successful available connection");
    }
}
