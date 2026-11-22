package task;

// Newly authored interface stub matching the published task API.
public abstract class ConnectionAPI {
    public static ConnectionAPI API;
    public abstract boolean isWifiAvailable();
    public abstract boolean isMobileDataAvailable();
}
