package task;

import java.util.ArrayList;

// Newly authored interface stub matching the published task API.
public abstract class DeviceAPI {
    public static DeviceAPI API;
    public abstract ArrayList<String> getAvailableConnections();
}
